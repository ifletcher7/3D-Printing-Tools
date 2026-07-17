bl_info = {
    "name": "Print Scale Limiter",
    "author": "OpenAI",
    "version": (1, 6, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Print Prep",
    "description": "Scale assemblies and replace selected structural members with oriented printable boxes",
    "category": "Object",
}

import bpy
import bmesh
import numpy as np
from collections import defaultdict
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from mathutils import Matrix, Vector


def get_pivot(context, mode):
    if mode == 'CURSOR':
        return context.scene.cursor.location.copy()
    return Vector((0.0, 0.0, 0.0))


def top_parent(obj):
    root = obj
    while root.parent is not None:
        root = root.parent
    return root


def hierarchy_objects(root):
    result = [root]
    stack = list(root.children)
    while stack:
        obj = stack.pop()
        result.append(obj)
        stack.extend(obj.children)
    return result


def parent_depth(obj):
    depth = 0
    current = obj.parent
    while current is not None:
        depth += 1
        current = current.parent
    return depth


def scale_matrix_about_pivot(scale, pivot):
    return (
        Matrix.Translation(pivot)
        @ Matrix.Scale(scale, 4)
        @ Matrix.Translation(-pivot)
    )


def evaluated_world_vertices(obj, depsgraph):
    """Return evaluated mesh vertices in world space."""
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        matrix_world = evaluated.matrix_world
        return np.array(
            [tuple(matrix_world @ vertex.co) for vertex in mesh.vertices],
            dtype=np.float64,
        )
    finally:
        evaluated.to_mesh_clear()


def pca_oriented_box(points):
    """
    Calculate a PCA-oriented bounding box.

    Returns:
        center_world: ndarray shape (3,)
        axes_world: ndarray shape (3, 3), axes stored as columns
        dimensions: ndarray shape (3,)
    """
    centroid = points.mean(axis=0)
    centered = points - centroid

    covariance = np.cov(centered, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Longest/most significant principal axis first.
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]

    # Ensure a right-handed coordinate system.
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0

    local_points = centered @ axes
    local_min = local_points.min(axis=0)
    local_max = local_points.max(axis=0)
    dimensions = local_max - local_min
    local_center = (local_min + local_max) * 0.5
    center_world = centroid + axes @ local_center

    return center_world, axes, dimensions


def create_box_mesh(name, dimensions):
    hx, hy, hz = (float(value) * 0.5 for value in dimensions)

    vertices = [
        (-hx, -hy, -hz), (hx, -hy, -hz),
        (hx,  hy, -hz), (-hx,  hy, -hz),
        (-hx, -hy,  hz), (hx, -hy,  hz),
        (hx,  hy,  hz), (-hx,  hy,  hz),
    ]

    faces = [
        (0, 1, 2, 3),
        (4, 7, 6, 5),
        (0, 4, 5, 1),
        (1, 5, 6, 2),
        (2, 6, 7, 3),
        (4, 0, 3, 7),
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def copy_materials(source, target):
    for slot in source.material_slots:
        if slot.material is not None:
            target.data.materials.append(slot.material)


class OBJECT_OT_print_scale_limiter(bpy.types.Operator):
    bl_idname = "object.print_scale_limiter"
    bl_label = "Scale Model Exactly"
    bl_description = (
        "Uniformly scale the complete model around one shared pivot without "
        "changing relative positions"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT'

    def execute(self, context):
        scene = context.scene
        scale_factor = scene.psl_scale_factor
        selected_only = scene.psl_selected_only
        include_hidden = scene.psl_include_hidden
        pivot = get_pivot(context, scene.psl_pivot)

        if scale_factor <= 0.0:
            self.report({'ERROR'}, "Scale factor must be greater than zero")
            return {'CANCELLED'}

        source = (
            list(context.selected_objects)
            if selected_only
            else list(scene.objects)
        )

        objects = [
            obj for obj in source
            if include_hidden or not obj.hide_get()
        ]

        if not objects:
            self.report({'ERROR'}, "No objects found to scale")
            return {'CANCELLED'}

        transform = scale_matrix_about_pivot(scale_factor, pivot)

        # Snapshot all world matrices before modifying anything. This prevents
        # parented children from inheriting the same scale twice.
        snapshots = [(obj, obj.matrix_world.copy()) for obj in objects]
        snapshots.sort(key=lambda item: parent_depth(item[0]))

        desired = [
            (obj, transform @ original_world)
            for obj, original_world in snapshots
        ]

        for obj, desired_world in desired:
            obj.matrix_world = desired_world

        context.view_layer.update()

        self.report(
            {'INFO'},
            f"Scaled {len(objects)} object(s) uniformly by {scale_factor:g}"
        )
        return {'FINISHED'}

def mm_to_blender_units(mm, scene):
    unit_scale = scene.unit_settings.scale_length
    if abs(unit_scale) <= 1.0e-12:
        unit_scale = 1.0
    return (mm / 1000.0) / unit_scale


def get_fallback_basis(points):
    center = Vector((0.0, 0.0, 0.0))
    for point in points:
        center += point
    center /= len(points)

    sample = points
    if len(sample) > 600:
        step = max(1, len(sample) // 600)
        sample = sample[::step]

    best_distance = 0.0
    axis_0 = Vector((1.0, 0.0, 0.0))

    for index, point in enumerate(sample):
        for other in sample[index + 1:]:
            delta = other - point
            distance = delta.length_squared
            if distance > best_distance:
                best_distance = distance
                if distance > 0.0:
                    axis_0 = delta.normalized()

    reference = Vector((0.0, 0.0, 1.0))
    if abs(axis_0.dot(reference)) > 0.9:
        reference = Vector((0.0, 1.0, 0.0))

    axis_1 = axis_0.cross(reference).normalized()
    axis_2 = axis_0.cross(axis_1).normalized()
    return center, [axis_0, axis_1, axis_2]


def get_member_basis(points):
    if np is None or len(points) < 3:
        return get_fallback_basis(points)

    array = np.array(
        [[point.x, point.y, point.z] for point in points],
        dtype=np.float64,
    )
    center_np = array.mean(axis=0)
    centered = array - center_np
    covariance = centered.T @ centered / max(len(array) - 1, 1)

    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    axes_np = vectors[:, order].T

    center = Vector(tuple(center_np.tolist()))
    axes = [
        Vector(tuple(axes_np[index].tolist())).normalized()
        for index in range(3)
    ]

    if axes[0].cross(axes[1]).dot(axes[2]) < 0.0:
        axes[2].negate()

    return center, axes


def measure_axis_spans(points, center, axes):
    coordinates = []
    minimums = [float("inf")] * 3
    maximums = [float("-inf")] * 3

    for point in points:
        offset = point - center
        coordinate = [offset.dot(axis) for axis in axes]
        coordinates.append(coordinate)

        for axis_index in range(3):
            minimums[axis_index] = min(
                minimums[axis_index],
                coordinate[axis_index],
            )
            maximums[axis_index] = max(
                maximums[axis_index],
                coordinate[axis_index],
            )

    spans = [
        maximums[index] - minimums[index]
        for index in range(3)
    ]
    return coordinates, spans


class OBJECT_OT_enforce_member_thickness(bpy.types.Operator):
    bl_idname = "object.enforce_member_thickness"
    bl_label = "Enforce Member Thickness"
    bl_description = (
        "Keep each selected member's long axis unchanged and enlarge only its "
        "two short PCA axes to the minimum printable thickness"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            context.mode == 'OBJECT'
            and any(obj.type == 'MESH' for obj in context.selected_objects)
        )

    def execute(self, context):
        scene = context.scene
        target = mm_to_blender_units(scene.psl_member_target_mm, scene)
        grow_only = scene.psl_member_grow_only
        minimum_slenderness = scene.psl_member_minimum_slenderness
        make_single_user = scene.psl_member_make_single_user

        selected = [
            obj for obj in context.selected_objects
            if obj.type == 'MESH' and not obj.hide_get()
        ]

        changed_count = 0
        unchanged_count = 0
        skipped_count = 0

        for obj in selected:
            if obj.data is None or len(obj.data.vertices) < 2:
                skipped_count += 1
                continue

            if make_single_user and obj.data.users > 1:
                obj.data = obj.data.copy()

            matrix_world = obj.matrix_world.copy()
            inverse_world = matrix_world.inverted_safe()
            points = [matrix_world @ vertex.co for vertex in obj.data.vertices]

            center, axes = get_member_basis(points)
            coordinates, spans = measure_axis_spans(points, center, axes)

            if any(span <= 1.0e-12 for span in spans):
                skipped_count += 1
                continue

            longest = spans[0]
            second = spans[1]
            slenderness = longest / max(second, 1.0e-12)

            if slenderness < minimum_slenderness:
                skipped_count += 1
                continue

            scales = [1.0, 1.0, 1.0]

            for axis_index in (1, 2):
                wanted = target / spans[axis_index]
                scales[axis_index] = (
                    max(1.0, wanted)
                    if grow_only
                    else wanted
                )

            if not any(abs(scale - 1.0) > 1.0e-6 for scale in scales):
                unchanged_count += 1
                continue

            for vertex, coordinate in zip(obj.data.vertices, coordinates):
                new_world = center.copy()

                for axis_index in range(3):
                    new_world += (
                        axes[axis_index]
                        * coordinate[axis_index]
                        * scales[axis_index]
                    )

                vertex.co = inverse_world @ new_world

            obj.data.update()
            changed_count += 1

        context.view_layer.update()

        self.report(
            {'INFO'},
            f"Adjusted {changed_count} member(s); "
            f"{unchanged_count} already met the target; "
            f"skipped {skipped_count}"
        )
        return {'FINISHED'}


class OBJECT_OT_replace_selected_with_boxes(bpy.types.Operator):
    bl_idname = "object.replace_selected_with_boxes"
    bl_label = "Replace Selected with Boxes"
    bl_description = (
        "Replace each selected mesh with a PCA-oriented box aligned to the actual "
        "member geometry, including diagonal beams"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            context.mode == 'OBJECT'
            and any(obj.type == 'MESH' for obj in context.selected_objects)
        )

    def execute(self, context):
        scene = context.scene
        minimum_size = scene.psl_box_minimum_size
        clamp_cross_section = scene.psl_box_clamp_cross_section
        keep_originals = scene.psl_box_keep_originals
        minimum_slenderness = scene.psl_box_minimum_slenderness

        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']
        depsgraph = context.evaluated_depsgraph_get()

        created = []
        skipped = []

        for source in selected:
            try:
                points = evaluated_world_vertices(source, depsgraph)
            except Exception as exc:
                skipped.append((source.name, str(exc)))
                continue

            if len(points) < 4:
                skipped.append((source.name, "not enough vertices"))
                continue

            center, axes, dimensions = pca_oriented_box(points)

            if np.any(dimensions <= 1.0e-9):
                skipped.append((source.name, "flat or zero-size geometry"))
                continue

            order = np.argsort(dimensions)[::-1]
            dimensions = dimensions[order]
            axes = axes[:, order]

            # Re-establish right-handed axes after sorting.
            if np.linalg.det(axes) < 0.0:
                axes[:, 2] *= -1.0

            longest = float(dimensions[0])
            second = float(dimensions[1])
            slenderness = longest / max(second, 1.0e-12)

            # Avoid turning plates, assemblies, or irregular clusters into huge blocks.
            if slenderness < minimum_slenderness:
                skipped.append((
                    source.name,
                    f"slenderness {slenderness:.2f} below threshold"
                ))
                continue

            if clamp_cross_section:
                dimensions[1] = max(float(dimensions[1]), minimum_size)
                dimensions[2] = max(float(dimensions[2]), minimum_size)

            mesh = create_box_mesh(f"{source.name}_BOX_MESH", dimensions)
            replacement = bpy.data.objects.new(f"{source.name}_BOX", mesh)

            # PCA axes are world-space basis vectors stored as columns.
            rotation = Matrix((
                tuple(axes[:, 0]),
                tuple(axes[:, 1]),
                tuple(axes[:, 2]),
            )).transposed().to_4x4()

            replacement.matrix_world = Matrix.Translation(Vector(center)) @ rotation

            for collection in source.users_collection:
                collection.objects.link(replacement)

            copy_materials(source, replacement)

            if keep_originals:
                source.hide_set(True)
                source.hide_render = True
            else:
                bpy.data.objects.remove(source, do_unlink=True)

            created.append(replacement)

        bpy.ops.object.select_all(action='DESELECT')
        for obj in created:
            obj.select_set(True)
        if created:
            context.view_layer.objects.active = created[0]

        message = f"Created {len(created)} oriented box member(s)"
        if skipped:
            message += f"; skipped {len(skipped)} object(s)"
        self.report({'INFO'}, message)

        if not created:
            self.report({
                'WARNING'
            }, "Nothing was replaced. Select individual slender beam objects.")
            return {'CANCELLED'}

        return {'FINISHED'}


class OBJECT_OT_flatten_selected_bottom(bpy.types.Operator):
    bl_idname = "object.flatten_selected_bottom"
    bl_label = "Flatten Selected Bottom"
    bl_description = (
        "Cut selected mesh objects at a horizontal Z plane, remove geometry below "
        "the plane, and optionally cap the cut"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            context.mode == 'OBJECT'
            and any(obj.type == 'MESH' for obj in context.selected_objects)
        )

    def execute(self, context):
        scene = context.scene
        selected = [obj for obj in context.selected_objects if obj.type == 'MESH']

        if scene.psl_flatten_height_source == 'CURSOR':
            cut_z = context.scene.cursor.location.z
        else:
            cut_z = scene.psl_flatten_z

        tolerance = scene.psl_flatten_tolerance
        fill_cut = scene.psl_flatten_fill

        processed = 0
        skipped = 0

        world_plane_point = Vector((0.0, 0.0, cut_z))
        world_plane_normal = Vector((0.0, 0.0, 1.0))

        for obj in selected:
            if obj.data is None or len(obj.data.vertices) == 0:
                skipped += 1
                continue

            # Make the mesh single-user before destructive editing.
            if obj.data.users > 1:
                obj.data = obj.data.copy()

            inverse_world = obj.matrix_world.inverted_safe()
            local_plane_point = inverse_world @ world_plane_point

            # For a world-space plane transformed into local coordinates,
            # the plane normal is transformed by M^T.
            local_plane_normal = (
                obj.matrix_world.to_3x3().transposed() @ world_plane_normal
            ).normalized()

            bm = bmesh.new()
            try:
                bm.from_mesh(obj.data)
                bm.verts.ensure_lookup_table()
                bm.edges.ensure_lookup_table()
                bm.faces.ensure_lookup_table()

                result = bmesh.ops.bisect_plane(
                    bm,
                    geom=list(bm.verts) + list(bm.edges) + list(bm.faces),
                    dist=tolerance,
                    plane_co=local_plane_point,
                    plane_no=local_plane_normal,
                    clear_inner=True,
                    clear_outer=False,
                )

                if fill_cut:
                    cut_edges = [
                        element for element in result.get("geom_cut", [])
                        if isinstance(element, bmesh.types.BMEdge)
                        and element.is_valid
                        and len(element.link_faces) < 2
                    ]

                    if cut_edges:
                        try:
                            bmesh.ops.holes_fill(
                                bm,
                                edges=cut_edges,
                                sides=0,
                            )
                        except Exception:
                            # Some imported meshes contain branching or duplicate
                            # cut edges that cannot be capped automatically.
                            pass

                bmesh.ops.remove_doubles(
                    bm,
                    verts=list(bm.verts),
                    dist=max(tolerance, 1.0e-9),
                )

                bm.normal_update()
                bm.to_mesh(obj.data)
                obj.data.update()
                processed += 1
            finally:
                bm.free()

        context.view_layer.update()

        message = (
            f"Flattened {processed} selected mesh object(s) at Z = {cut_z:.6g}"
        )
        if skipped:
            message += f"; skipped {skipped} empty mesh(es)"

        self.report({'INFO'}, message)
        return {'FINISHED'}


class VIEW3D_PT_print_scale_limiter(bpy.types.Panel):
    bl_label = "Print Scale Limiter"
    bl_idname = "VIEW3D_PT_print_scale_limiter"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Print Prep"

    def draw(self, context):
        layout = self.layout
        scene = context.scene

        column = layout.column(align=True)
        column.prop(scene, "psl_scale_factor")
        column.prop(scene, "psl_pivot")

        layout.separator()

        column = layout.column(align=True)
        column.prop(scene, "psl_selected_only")
        column.prop(scene, "psl_include_hidden")

        layout.separator()
        layout.operator("object.print_scale_limiter", icon='MOD_LENGTH')

        box = layout.box()
        box.label(text="Applies one exact uniform scale.")
        box.label(text="Relative positions remain unchanged.")
        box.label(text="Use 0.005 for a 1:200 model.")

        layout.separator()

        box = layout.box()
        box.label(text="Minimum Member Thickness")
        box.prop(scene, "psl_member_target_mm")
        box.prop(scene, "psl_member_grow_only")
        box.prop(scene, "psl_member_minimum_slenderness")
        box.prop(scene, "psl_member_make_single_user")
        box.operator("object.enforce_member_thickness", icon='MOD_LENGTH')
        box.label(text="Keeps the longest axis unchanged.")
        box.label(text="Select individual straight members only.")

        layout.separator()

        box = layout.box()
        box.label(text="Beam / Member Simplification")
        box.prop(scene, "psl_box_clamp_cross_section")
        if scene.psl_box_clamp_cross_section:
            box.prop(scene, "psl_box_minimum_size")
        box.prop(scene, "psl_box_minimum_slenderness")
        box.prop(scene, "psl_box_keep_originals")
        box.operator("object.replace_selected_with_boxes", icon='MESH_CUBE')
        box.label(text="Uses PCA to align diagonal members.")
        box.label(text="Select individual straight members only.")

        layout.separator()

        box = layout.box()
        box.label(text="Flatten Bottom")
        box.prop(scene, "psl_flatten_height_source")
        if scene.psl_flatten_height_source == 'NUMERIC':
            box.prop(scene, "psl_flatten_z")
        box.prop(scene, "psl_flatten_fill")
        box.prop(scene, "psl_flatten_tolerance")
        box.operator("object.flatten_selected_bottom", icon='MOD_BOOLEAN')
        box.label(text="Only selected mesh objects are cut.")
        box.label(text="Leave intentional lower parts unselected.")


classes = (
    OBJECT_OT_print_scale_limiter,
    OBJECT_OT_enforce_member_thickness,
    OBJECT_OT_replace_selected_with_boxes,
    OBJECT_OT_flatten_selected_bottom,
    VIEW3D_PT_print_scale_limiter,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.psl_scale_factor = FloatProperty(
        name="Scale Factor",
        description="Requested uniform model scale; use 0.005 for 1:200",
        default=0.005,
        min=1.0e-8,
        soft_max=1.0,
        precision=6,
    )

    bpy.types.Scene.psl_minimum_size = FloatProperty(
        name="Minimum Dimension",
        description="Smallest permitted final mesh-object dimension",
        default=0.002,
        min=1.0e-8,
        subtype='DISTANCE',
        unit='LENGTH',
        precision=4,
    )

    bpy.types.Scene.psl_grouping = EnumProperty(
        name="Assembly Grouping",
        description="Choose which objects must remain rigidly positioned together",
        items=(
            ('WHOLE_MODEL', "Whole Model", "Use one scale for the entire model"),
            ('ROOT_PARENT', "Root Parent", "Use one scale for each top-level hierarchy"),
        ),
        default='WHOLE_MODEL',
    )

    bpy.types.Scene.psl_pivot = EnumProperty(
        name="Scale Pivot",
        description="Shared pivot used to scale each complete assembly",
        items=(
            ('WORLD', "World Origin", "Scale around the world origin"),
            ('CURSOR', "3D Cursor", "Scale around the 3D cursor"),
        ),
        default='WORLD',
    )

    bpy.types.Scene.psl_selected_only = BoolProperty(
        name="Selected Objects Only",
        description="Process selected mesh objects instead of every mesh object",
        default=False,
    )

    bpy.types.Scene.psl_include_hidden = BoolProperty(
        name="Include Hidden Objects",
        description="Process hidden objects",
        default=False,
    )

    bpy.types.Scene.psl_box_minimum_size = FloatProperty(
        name="Minimum Cross-Section",
        description="Minimum width and depth for replacement box members",
        default=0.002,
        min=1.0e-8,
        subtype='DISTANCE',
        unit='LENGTH',
        precision=4,
    )

    bpy.types.Scene.psl_box_clamp_cross_section = BoolProperty(
        name="Clamp Cross-Section",
        description="Increase the two short box dimensions to the minimum size",
        default=True,
    )

    bpy.types.Scene.psl_box_keep_originals = BoolProperty(
        name="Hide Originals",
        description="Hide original members instead of deleting them",
        default=True,
    )

    bpy.types.Scene.psl_box_minimum_slenderness = FloatProperty(
        name="Minimum Slenderness",
        description=(
            "Only replace objects whose longest dimension divided by their "
            "second-longest dimension meets this value"
        ),
        default=3.0,
        min=1.0,
        soft_max=20.0,
        precision=2,
    )

    bpy.types.Scene.psl_member_target_mm = FloatProperty(
        name="Target Thickness (mm)",
        description="Minimum width and depth for selected straight members",
        default=2.0,
        min=0.001,
        soft_max=20.0,
        precision=3,
    )

    bpy.types.Scene.psl_member_grow_only = BoolProperty(
        name="Grow Only",
        description="Leave dimensions above the target unchanged",
        default=True,
    )

    bpy.types.Scene.psl_member_minimum_slenderness = FloatProperty(
        name="Minimum Slenderness",
        description=(
            "Skip objects whose longest PCA span is not sufficiently larger "
            "than the second-longest span"
        ),
        default=3.0,
        min=1.0,
        soft_max=20.0,
        precision=2,
    )

    bpy.types.Scene.psl_member_make_single_user = BoolProperty(
        name="Make Mesh Single-User",
        description=(
            "Copy shared mesh data before resizing so linked duplicates are "
            "not changed together"
        ),
        default=True,
    )

    bpy.types.Scene.psl_flatten_height_source = EnumProperty(
        name="Cut Height",
        description="Choose where the horizontal bottom-cut plane is placed",
        items=(
            ('CURSOR', "3D Cursor Z", "Use the current 3D cursor Z position"),
            ('NUMERIC', "Numeric Z", "Use the numeric Z value below"),
        ),
        default='CURSOR',
    )

    bpy.types.Scene.psl_flatten_z = FloatProperty(
        name="Cut Z",
        description="World-space Z height of the flat bottom",
        default=0.0,
        subtype='DISTANCE',
        unit='LENGTH',
        precision=4,
    )

    bpy.types.Scene.psl_flatten_fill = BoolProperty(
        name="Cap Cut Surface",
        description="Attempt to fill the new horizontal boundary after cutting",
        default=True,
    )

    bpy.types.Scene.psl_flatten_tolerance = FloatProperty(
        name="Cut Tolerance",
        description="Tolerance used by the bisect and vertex cleanup operations",
        default=0.00001,
        min=1.0e-9,
        soft_max=0.001,
        subtype='DISTANCE',
        unit='LENGTH',
        precision=6,
    )


def unregister():
    property_names = (
        "psl_member_make_single_user",
        "psl_member_minimum_slenderness",
        "psl_member_grow_only",
        "psl_member_target_mm",
        "psl_flatten_tolerance",
        "psl_flatten_fill",
        "psl_flatten_z",
        "psl_flatten_height_source",
        "psl_box_minimum_slenderness",
        "psl_box_keep_originals",
        "psl_box_clamp_cross_section",
        "psl_box_minimum_size",
        "psl_include_hidden",
        "psl_selected_only",
        "psl_pivot",
        "psl_grouping",
        "psl_minimum_size",
        "psl_scale_factor",
    )

    for name in property_names:
        if hasattr(bpy.types.Scene, name):
            delattr(bpy.types.Scene, name)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()