bl_info = {
    "name": "Print Section Splitter",
    "author": "OpenAI",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Scene  Divider",
    "description": "Split selected mesh geometry into configurable print-volume sections",
    "category": "Object",
}

import bpy
import bmesh
import math
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from mathutils import Vector

EPSILON = 1.0e-8


def selected_world_bounds(objects, depsgraph):
    minimum = Vector((float("inf"),) * 3)
    maximum = Vector((float("-inf"),) * 3)
    found = False
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        matrix = evaluated.matrix_world
        for corner in evaluated.bound_box:
            point = matrix @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], point[axis])
                maximum[axis] = max(maximum[axis], point[axis])
            found = True
    return (minimum, maximum) if found else (None, None)


def grid_start(value, origin, size):
    return origin + math.floor((value - origin) / size) * size


def grid_end(value, origin, size):
    return origin + math.ceil((value - origin) / size) * size


def fill_cut_edges(bm, cut_geometry):
    edges = [
        element for element in cut_geometry
        if isinstance(element, bmesh.types.BMEdge)
        and element.is_valid
        and len(element.link_faces) < 2
    ]
    if not edges:
        return
    try:
        bmesh.ops.holes_fill(bm, edges=edges, sides=0)
    except Exception:
        pass


def clip_bmesh_to_box(bm, box_min, box_max, tolerance, cap_cuts):
    planes = (
        (Vector((box_min.x, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), True),
        (Vector((box_max.x, 0.0, 0.0)), Vector((1.0, 0.0, 0.0)), False),
        (Vector((0.0, box_min.y, 0.0)), Vector((0.0, 1.0, 0.0)), True),
        (Vector((0.0, box_max.y, 0.0)), Vector((0.0, 1.0, 0.0)), False),
        (Vector((0.0, 0.0, box_min.z)), Vector((0.0, 0.0, 1.0)), True),
        (Vector((0.0, 0.0, box_max.z)), Vector((0.0, 0.0, 1.0)), False),
    )

    for point, normal, keep_positive in planes:
        geometry = list(bm.verts) + list(bm.edges) + list(bm.faces)
        if not geometry:
            return False
        result = bmesh.ops.bisect_plane(
            bm,
            geom=geometry,
            dist=tolerance,
            plane_co=point,
            plane_no=normal,
            clear_inner=keep_positive,
            clear_outer=not keep_positive,
        )
        if cap_cuts:
            fill_cut_edges(bm, result.get("geom_cut", []))
        if not bm.verts:
            return False

    bmesh.ops.remove_doubles(
        bm,
        verts=list(bm.verts),
        dist=max(tolerance, EPSILON),
    )
    loose_verts = [vert for vert in bm.verts if not vert.link_edges]
    if loose_verts:
        bmesh.ops.delete(bm, geom=loose_verts, context='VERTS')
    bm.normal_update()
    return bool(bm.faces or bm.edges)


def build_combined_source(objects, depsgraph):
    combined_mesh = bpy.data.meshes.new("PrintSection_SourceMesh")
    combined_bm = bmesh.new()
    combined_materials = []
    material_lookup = {}
    try:
        for obj in objects:
            evaluated = obj.evaluated_get(depsgraph)
            temp_mesh = evaluated.to_mesh()
            temp_bm = bmesh.new()
            try:
                temp_bm.from_mesh(temp_mesh)
                temp_bm.transform(evaluated.matrix_world)

                index_map = {}
                for index, material in enumerate(temp_mesh.materials):
                    if material is None:
                        index_map[index] = 0
                        continue
                    key = material.as_pointer()
                    if key not in material_lookup:
                        material_lookup[key] = len(combined_materials)
                        combined_materials.append(material)
                    index_map[index] = material_lookup[key]

                for face in temp_bm.faces:
                    face.material_index = index_map.get(face.material_index, 0)

                temp_copy = bpy.data.meshes.new("PrintSection_Temp")
                temp_bm.to_mesh(temp_copy)
                combined_bm.from_mesh(temp_copy)
                bpy.data.meshes.remove(temp_copy)
            finally:
                temp_bm.free()
                evaluated.to_mesh_clear()

        combined_bm.to_mesh(combined_mesh)
        combined_mesh.update()
        for material in combined_materials:
            combined_mesh.materials.append(material)
        return combined_mesh
    finally:
        combined_bm.free()


class OBJECT_OT_split_into_print_sections(bpy.types.Operator):
    bl_idname = "object.split_into_print_sections"
    bl_label = "Split Selected into Sections"
    bl_description = "Split selected mesh geometry into X/Y/Z print-volume sections"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and any(
            obj.type == 'MESH' for obj in context.selected_objects
        )

    def execute(self, context):
        scene = context.scene
        selected = [
            obj for obj in context.selected_objects
            if obj.type == 'MESH' and not obj.hide_get()
        ]
        if not selected:
            self.report({'ERROR'}, "Select at least one visible mesh object")
            return {'CANCELLED'}

        size = Vector((scene.pss_size_x, scene.pss_size_y, scene.pss_size_z))
        if min(size) <= 0.0:
            self.report({'ERROR'}, "All section dimensions must be greater than zero")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        bounds_min, bounds_max = selected_world_bounds(selected, depsgraph)
        if bounds_min is None:
            self.report({'ERROR'}, "Could not calculate selected geometry bounds")
            return {'CANCELLED'}

        origin = (
            context.scene.cursor.location.copy()
            if scene.pss_grid_origin == 'CURSOR'
            else bounds_min.copy()
        )

        start = Vector((
            grid_start(bounds_min.x, origin.x, size.x),
            grid_start(bounds_min.y, origin.y, size.y),
            grid_start(bounds_min.z, origin.z, size.z),
        ))
        end = Vector((
            grid_end(bounds_max.x, origin.x, size.x),
            grid_end(bounds_max.y, origin.y, size.y),
            grid_end(bounds_max.z, origin.z, size.z),
        ))

        counts = (
            max(1, int(round((end.x - start.x) / size.x))),
            max(1, int(round((end.y - start.y) / size.y))),
            max(1, int(round((end.z - start.z) / size.z))),
        )
        total_cells = counts[0] * counts[1] * counts[2]
        if total_cells > scene.pss_max_cells:
            self.report(
                {'ERROR'},
                f"Grid would create {total_cells} cells; maximum is {scene.pss_max_cells}"
            )
            return {'CANCELLED'}

        source_mesh = build_combined_source(selected, depsgraph)
        collection_name = scene.pss_collection_name.strip() or "Print Sections"
        output_collection = bpy.data.collections.get(collection_name)
        if output_collection is None:
            output_collection = bpy.data.collections.new(collection_name)
            scene.collection.children.link(output_collection)

        created = []
        overlap = max(0.0, scene.pss_overlap)
        tolerance = scene.pss_tolerance

        try:
            for ix in range(counts[0]):
                for iy in range(counts[1]):
                    for iz in range(counts[2]):
                        cell_min = Vector((
                            start.x + ix * size.x - overlap,
                            start.y + iy * size.y - overlap,
                            start.z + iz * size.z - overlap,
                        ))
                        cell_max = Vector((
                            start.x + (ix + 1) * size.x + overlap,
                            start.y + (iy + 1) * size.y + overlap,
                            start.z + (iz + 1) * size.z + overlap,
                        ))

                        bm = bmesh.new()
                        try:
                            bm.from_mesh(source_mesh)
                            if not clip_bmesh_to_box(
                                bm,
                                cell_min,
                                cell_max,
                                tolerance,
                                scene.pss_cap_cuts,
                            ):
                                continue

                            name = f"Section_X{ix + 1:02d}_Y{iy + 1:02d}_Z{iz + 1:02d}"
                            mesh = bpy.data.meshes.new(f"{name}_Mesh")
                            bm.to_mesh(mesh)
                            mesh.update()
                            for material in source_mesh.materials:
                                mesh.materials.append(material)

                            obj = bpy.data.objects.new(name, mesh)
                            output_collection.objects.link(obj)
                            created.append(obj)
                        finally:
                            bm.free()
        finally:
            bpy.data.meshes.remove(source_mesh)

        if scene.pss_hide_originals:
            for obj in selected:
                obj.hide_set(True)
                obj.hide_render = True

        bpy.ops.object.select_all(action='DESELECT')
        for obj in created:
            obj.select_set(True)
        if created:
            context.view_layer.objects.active = created[0]

        self.report(
            {'INFO'},
            f"Created {len(created)} non-empty section(s) from "
            f"{counts[0]} × {counts[1]} × {counts[2]} grid cells"
        )
        return {'FINISHED'}


class VIEW3D_PT_print_section_splitter(bpy.types.Panel):
    bl_label = "Print Section Splitter"
    bl_idname = "VIEW3D_PT_print_section_splitter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Print Prep"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        col = layout.column(align=True)
        col.prop(scene, "pss_size_x")
        col.prop(scene, "pss_size_y")
        col.prop(scene, "pss_size_z")

        layout.separator()
        col = layout.column(align=True)
        col.prop(scene, "pss_grid_origin")
        col.prop(scene, "pss_overlap")
        col.prop(scene, "pss_cap_cuts")
        col.prop(scene, "pss_hide_originals")
        col.prop(scene, "pss_tolerance")

        layout.separator()
        layout.operator("object.split_into_print_sections", icon='MOD_BOOLEAN')

        box = layout.box()
        box.label(text="Default: 250 × 250 × 500 mm")
        box.label(text="X/Y horizontal; Z vertical.")
        box.label(text="Original objects remain unchanged.")


classes = (
    OBJECT_OT_split_into_print_sections,
    VIEW3D_PT_print_section_splitter,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.pss_size_x = FloatProperty(
        name="Section Width X", default=0.250, min=0.001,
        subtype='DISTANCE', unit='LENGTH', precision=4,
    )
    bpy.types.Scene.pss_size_y = FloatProperty(
        name="Section Depth Y", default=0.250, min=0.001,
        subtype='DISTANCE', unit='LENGTH', precision=4,
    )
    bpy.types.Scene.pss_size_z = FloatProperty(
        name="Section Height Z", default=0.500, min=0.001,
        subtype='DISTANCE', unit='LENGTH', precision=4,
    )
    bpy.types.Scene.pss_grid_origin = EnumProperty(
        name="Grid Origin",
        items=(
            ('BOUNDS_MIN', "Selection Minimum", "Start at selected minimum corner"),
            ('CURSOR', "3D Cursor", "Align grid to the 3D cursor"),
        ),
        default='BOUNDS_MIN',
    )
    bpy.types.Scene.pss_overlap = FloatProperty(
        name="Section Overlap", default=0.0, min=0.0, soft_max=0.005,
        subtype='DISTANCE', unit='LENGTH', precision=4,
    )
    bpy.types.Scene.pss_cap_cuts = BoolProperty(
        name="Cap Cut Surfaces", default=True,
    )
    bpy.types.Scene.pss_hide_originals = BoolProperty(
        name="Hide Originals", default=True,
    )
    bpy.types.Scene.pss_tolerance = FloatProperty(
        name="Cut Tolerance", default=0.00001, min=1.0e-9, soft_max=0.001,
        subtype='DISTANCE', unit='LENGTH', precision=6,
    )
    bpy.types.Scene.pss_max_cells = bpy.props.IntProperty(
        name="Maximum Grid Cells", default=500, min=1, soft_max=2000,
    )
    bpy.types.Scene.pss_collection_name = bpy.props.StringProperty(
        name="Output Collection", default="Print Sections",
    )


def unregister():
    for name in (
        "pss_collection_name", "pss_max_cells", "pss_tolerance",
        "pss_hide_originals", "pss_cap_cuts", "pss_overlap",
        "pss_grid_origin", "pss_size_z", "pss_size_y", "pss_size_x",
    ):
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()