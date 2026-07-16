bl_info = {
    "name": "3D Print Prep",
    "author": "OpenAI",
    "version": (2, 0, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > 3D Print Prep",
    "description": "Optionally replace meshes with solid boxes, then scale to print size",
    "category": "Object",
}

import bpy
from mathutils import Vector
from bpy.props import BoolProperty, FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

OUTPUT_COLLECTION = "3D_Print_Prep_Output"


def source_objects(context, selected_only):
    """Return visible mesh objects, optionally limited to the current selection."""
    objects = context.selected_objects if selected_only else context.scene.objects
    return [obj for obj in objects if obj.type == "MESH" and obj.visible_get()]


def mm_to_blender_units(scene, millimetres):
    """Convert physical millimetres to Blender units using the scene unit scale."""
    scale_length = scene.unit_settings.scale_length or 1.0
    return (millimetres / 1000.0) / scale_length


def world_bounds(objects):
    points = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
    ]
    if not points:
        return None, None

    return (
        Vector((
            min(point.x for point in points),
            min(point.y for point in points),
            min(point.z for point in points),
        )),
        Vector((
            max(point.x for point in points),
            max(point.y for point in points),
            max(point.z for point in points),
        )),
    )


def clear_output_collection(scene):
    collection = bpy.data.collections.get(OUTPUT_COLLECTION)
    if collection is None:
        collection = bpy.data.collections.new(OUTPUT_COLLECTION)
        scene.collection.children.link(collection)
        return collection

    for obj in list(collection.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    return collection


def evaluated_duplicate(context, source, collection):
    """Duplicate the evaluated mesh so existing modifiers are baked into the copy."""
    depsgraph = context.evaluated_depsgraph_get()
    evaluated = source.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated, depsgraph=depsgraph)

    duplicate = bpy.data.objects.new(f"{source.name}_PRINT", mesh)
    duplicate.matrix_world = source.matrix_world.copy()
    duplicate["print_prep_source"] = source.name
    collection.objects.link(duplicate)
    return duplicate


def replace_with_local_bounding_box(obj):
    """
    Replace the mesh with a closed rectangular solid matching its local bounds.

    This fills hollow I-beam/channel profiles while preserving their outer
    length, width, height, orientation, and object placement.
    """
    corners = [Vector(corner) for corner in obj.bound_box]
    minimum = Vector((
        min(corner.x for corner in corners),
        min(corner.y for corner in corners),
        min(corner.z for corner in corners),
    ))
    maximum = Vector((
        max(corner.x for corner in corners),
        max(corner.y for corner in corners),
        max(corner.z for corner in corners),
    ))

    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    vertices = (
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    )
    faces = (
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    )

    old_mesh = obj.data
    new_mesh = bpy.data.meshes.new(f"{obj.name}_SOLID")
    new_mesh.from_pydata(vertices, (), faces)
    new_mesh.validate()
    new_mesh.update()
    obj.data = new_mesh

    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)


def scale_about_pivot(objects, pivot, factor):
    """Uniformly scale object geometry and spacing around one common pivot."""
    for obj in objects:
        matrix = obj.matrix_world.copy()
        matrix.translation = pivot + (matrix.translation - pivot) * factor

        for axis in range(3):
            matrix.col[axis].xyz *= factor

        obj.matrix_world = matrix
        obj["print_prep_scale"] = factor


def local_world_dimensions(obj):
    """Return dimensions along the object's local axes in world units."""
    corners = [Vector(corner) for corner in obj.bound_box]
    local_size = Vector((
        max(c.x for c in corners) - min(c.x for c in corners),
        max(c.y for c in corners) - min(c.y for c in corners),
        max(c.z for c in corners) - min(c.z for c in corners),
    ))
    basis_lengths = Vector((
        obj.matrix_world.col[0].xyz.length,
        obj.matrix_world.col[1].xyz.length,
        obj.matrix_world.col[2].xyz.length,
    ))
    return Vector((
        local_size.x * basis_lengths.x,
        local_size.y * basis_lengths.y,
        local_size.z * basis_lengths.z,
    ))


def enforce_minimum_dimensions(obj, minimum_size):
    """
    Increase only deficient local axes to the minimum printable size.

    The object centre and orientation stay unchanged. This intentionally alters
    proportions when a scaled beam would otherwise be too thin to print.
    """
    dimensions = local_world_dimensions(obj)
    corrections = Vector((
        minimum_size / dimensions.x if 0.0 < dimensions.x < minimum_size else 1.0,
        minimum_size / dimensions.y if 0.0 < dimensions.y < minimum_size else 1.0,
        minimum_size / dimensions.z if 0.0 < dimensions.z < minimum_size else 1.0,
    ))

    changed = False
    matrix = obj.matrix_world.copy()

    for axis, correction in enumerate(corrections):
        if correction > 1.0:
            matrix.col[axis].xyz *= correction
            changed = True

    if changed:
        obj.matrix_world = matrix
    return changed


class PRINTPREP_Settings(PropertyGroup):
    selected_only: BoolProperty(
        name="Selected Objects Only",
        default=False,
        description="Process selected visible meshes instead of every visible mesh",
    )
    make_solid_boxes: BoolProperty(
        name="Fill Beams as Solid Boxes",
        default=True,
        description=(
            "Replace each mesh with a closed rectangular solid matching its "
            "outer local dimensions"
        ),
    )
    scale_denominator: FloatProperty(
        name="Scale Denominator",
        default=200.0,
        min=1.0,
        description="A value of 200 produces a 1:200 model",
    )
    minimum_size_mm: FloatProperty(
        name="Minimum Dimension",
        default=2.0,
        min=0.01,
        precision=3,
        description="Any local object dimension below this value is enlarged to it",
    )
    keep_originals: BoolProperty(
        name="Keep Original Objects",
        default=True,
        description="Create processed duplicates and leave the source model untouched",
    )


class PRINTPREP_OT_prepare(Operator):
    bl_idname = "printprep.prepare"
    bl_label = "Prepare for 3D Printing"
    bl_description = "Fill beams, scale the model, and enforce minimum dimensions"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        settings = context.scene.printprep_settings
        sources = source_objects(context, settings.selected_only)

        if not sources:
            self.report({"ERROR"}, "No visible mesh objects found")
            return {"CANCELLED"}

        if settings.keep_originals:
            collection = clear_output_collection(context.scene)
            objects = [
                evaluated_duplicate(context, source, collection)
                for source in sources
            ]
        else:
            objects = sources

        if settings.make_solid_boxes:
            for obj in objects:
                replace_with_local_bounding_box(obj)

        minimum, maximum = world_bounds(objects)
        if minimum is None:
            self.report({"ERROR"}, "Unable to calculate model bounds")
            return {"CANCELLED"}

        pivot = (minimum + maximum) * 0.5
        factor = 1.0 / settings.scale_denominator
        scale_about_pivot(objects, pivot, factor)

        minimum_size = mm_to_blender_units(
            context.scene,
            settings.minimum_size_mm,
        )
        adjusted = sum(
            enforce_minimum_dimensions(obj, minimum_size)
            for obj in objects
        )

        for obj in objects:
            obj["print_prep_minimum_mm"] = settings.minimum_size_mm

        self.report(
            {"INFO"},
            (
                f"Prepared {len(objects)} objects at 1:{settings.scale_denominator:g}; "
                f"{adjusted} objects required minimum-size correction"
            ),
        )
        return {"FINISHED"}


class PRINTPREP_PT_panel(Panel):
    bl_label = "3D Print Prep"
    bl_idname = "PRINTPREP_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "3D Print Prep"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.printprep_settings

        layout.prop(settings, "selected_only")
        layout.prop(settings, "keep_originals")
        layout.separator()
        layout.prop(settings, "make_solid_boxes")
        layout.prop(settings, "scale_denominator")
        layout.prop(settings, "minimum_size_mm")
        layout.separator()
        layout.operator("printprep.prepare", icon="MOD_REMESH")


CLASSES = (
    PRINTPREP_Settings,
    PRINTPREP_OT_prepare,
    PRINTPREP_PT_panel,
)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.printprep_settings = PointerProperty(type=PRINTPREP_Settings)


def unregister():
    del bpy.types.Scene.printprep_settings
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()