bl_info = {
    "name": "Beam to Solid Rectangular Prism",
    "author": "OpenAI",
    "version": (1, 1, 0),
    "blender": (4, 0, 0),
    "location": "3D Viewport > Sidebar > 3D Print",
    "description": "Replaces beam meshes with tightly fitted, oriented solid rectangular prisms",
    "category": "Object",
}

import bpy
from math import cos, radians
from mathutils import Vector
from bpy.props import BoolProperty, EnumProperty, FloatProperty
from bpy.types import Operator, Panel


def get_mesh_data(obj, depsgraph, use_evaluated):
    source_obj = obj.evaluated_get(depsgraph) if use_evaluated else obj
    mesh = source_obj.to_mesh() if use_evaluated else obj.data
    return source_obj, mesh


def release_mesh(source_obj, use_evaluated):
    if use_evaluated:
        source_obj.to_mesh_clear()


def canonical_direction(direction):
    """Treat opposite directions as the same axis."""
    direction = direction.normalized()
    for component in direction:
        if abs(component) > 1e-10:
            if component < 0.0:
                direction.negate()
            break
    return direction


def cluster_face_normals(mesh, angle_tolerance_degrees):
    """
    Cluster polygon normals, ignoring sign.

    Face normals are preferable to mesh-edge directions because triangulation
    adds diagonal edges, while the surface normals of an extruded steel member
    still identify its true orthogonal axes.
    """
    threshold = cos(radians(angle_tolerance_degrees))
    clusters = []

    for polygon in mesh.polygons:
        if polygon.area <= 1e-12:
            continue

        normal = canonical_direction(polygon.normal.copy())
        weight = polygon.area

        best_cluster = None
        best_dot = threshold

        for cluster in clusters:
            similarity = abs(normal.dot(cluster["axis"]))
            if similarity >= best_dot:
                best_dot = similarity
                best_cluster = cluster

        if best_cluster is None:
            clusters.append({
                "sum": normal * weight,
                "weight": weight,
                "axis": normal,
            })
        else:
            if normal.dot(best_cluster["axis"]) < 0.0:
                normal.negate()
            best_cluster["sum"] += normal * weight
            best_cluster["weight"] += weight
            best_cluster["axis"] = best_cluster["sum"].normalized()

    clusters.sort(key=lambda item: item["weight"], reverse=True)
    return clusters


def find_orthogonal_axes_from_faces(mesh, angle_tolerance_degrees):
    clusters = cluster_face_normals(mesh, angle_tolerance_degrees)
    if len(clusters) < 2:
        return None

    # Search the strongest face-normal clusters for the best perpendicular pair.
    max_dot = abs(cos(radians(90.0 - angle_tolerance_degrees)))
    candidates = clusters[:24]

    best_pair = None
    best_score = -1.0

    for index, first in enumerate(candidates):
        for second in candidates[index + 1:]:
            dot_value = abs(first["axis"].dot(second["axis"]))
            if dot_value <= max_dot:
                score = first["weight"] + second["weight"]
                if score > best_score:
                    best_score = score
                    best_pair = (first["axis"].copy(), second["axis"].copy())

    if best_pair is None:
        return None

    axis_x = best_pair[0].normalized()

    # Gram-Schmidt removes tiny import/rounding errors without changing the
    # intended beam direction.
    axis_y = best_pair[1] - axis_x * best_pair[1].dot(axis_x)
    if axis_y.length_squared <= 1e-16:
        return None
    axis_y.normalize()

    axis_z = axis_x.cross(axis_y)
    if axis_z.length_squared <= 1e-16:
        return None
    axis_z.normalize()

    # Recompute Y so the final basis is exactly orthonormal.
    axis_y = axis_z.cross(axis_x).normalized()
    return axis_x, axis_y, axis_z


def covariance_axes(mesh):
    """PCA fallback for meshes whose face normals do not provide a clean basis."""
    vertices = [vertex.co.copy() for vertex in mesh.vertices]
    if len(vertices) < 3:
        return None

    center = sum(vertices, Vector()) / len(vertices)

    xx = xy = xz = yy = yz = zz = 0.0
    for point in vertices:
        offset = point - center
        xx += offset.x * offset.x
        xy += offset.x * offset.y
        xz += offset.x * offset.z
        yy += offset.y * offset.y
        yz += offset.y * offset.z
        zz += offset.z * offset.z

    # Power iteration for the dominant eigenvector.
    def multiply(vector):
        return Vector((
            xx * vector.x + xy * vector.y + xz * vector.z,
            xy * vector.x + yy * vector.y + yz * vector.z,
            xz * vector.x + yz * vector.y + zz * vector.z,
        ))

    axis_x = Vector((1.0, 0.731, 0.417)).normalized()
    for _ in range(32):
        result = multiply(axis_x)
        if result.length_squared <= 1e-20:
            return None
        axis_x = result.normalized()

    # Find a second axis by power iteration constrained perpendicular to X.
    axis_y = Vector((0.317, 1.0, 0.613))
    axis_y -= axis_x * axis_y.dot(axis_x)
    axis_y.normalize()

    for _ in range(32):
        result = multiply(axis_y)
        result -= axis_x * result.dot(axis_x)
        if result.length_squared <= 1e-20:
            break
        axis_y = result.normalized()

    axis_z = axis_x.cross(axis_y)
    if axis_z.length_squared <= 1e-16:
        return None
    axis_z.normalize()
    axis_y = axis_z.cross(axis_x).normalized()

    return axis_x, axis_y, axis_z


def oriented_bounds(mesh, axes):
    vertices = mesh.vertices
    if not vertices:
        return None

    minimums = [float("inf")] * 3
    maximums = [float("-inf")] * 3

    for vertex in vertices:
        point = vertex.co
        for index, axis in enumerate(axes):
            projection = point.dot(axis)
            minimums[index] = min(minimums[index], projection)
            maximums[index] = max(maximums[index], projection)

    return minimums, maximums


def make_oriented_box_mesh(name, axes, minimums, maximums):
    centers = [(minimums[i] + maximums[i]) * 0.5 for i in range(3)]
    halves = [(maximums[i] - minimums[i]) * 0.5 for i in range(3)]

    center = (
        axes[0] * centers[0]
        + axes[1] * centers[1]
        + axes[2] * centers[2]
    )

    vertices = []
    for sx, sy, sz in (
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1),  (1, -1, 1),  (1, 1, 1),  (-1, 1, 1),
    ):
        vertices.append(
            center
            + axes[0] * (sx * halves[0])
            + axes[1] * (sy * halves[1])
            + axes[2] * (sz * halves[2])
        )

    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ]

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.validate(verbose=False)
    mesh.update(calc_edges=True)
    return mesh


def ensure_backup_collection(scene):
    name = "Beam Prism Backups"
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        scene.collection.children.link(collection)
    return collection


class OBJECT_OT_beams_to_prisms(Operator):
    bl_idname = "object.beams_to_prisms"
    bl_label = "Convert Beams to Prisms"
    bl_description = (
        "Replace each target mesh with a tightly fitted solid rectangular prism "
        "aligned to the beam's actual face directions"
    )
    bl_options = {"REGISTER", "UNDO"}

    target: EnumProperty(
        name="Objects",
        items=(
            ("SELECTED", "Selected Meshes", "Convert selected mesh objects only"),
            ("VISIBLE", "All Visible Meshes", "Convert every visible mesh object"),
        ),
        default="SELECTED",
    )

    use_evaluated: BoolProperty(
        name="Include Modifiers",
        description=(
            "Measure geometry after modifiers. Leave disabled when exact imported "
            "Navisworks dimensions are required"
        ),
        default=False,
    )

    keep_backups: BoolProperty(
        name="Keep Hidden Backups",
        description="Create hidden duplicates of the original objects",
        default=True,
    )

    preserve_materials: BoolProperty(
        name="Preserve Materials",
        description="Copy material slots to the replacement prism",
        default=True,
    )

    normal_tolerance: FloatProperty(
        name="Direction Tolerance",
        description="Angular tolerance used to group nearly parallel beam faces",
        default=1.0,
        min=0.05,
        max=10.0,
        subtype="ANGLE",
        unit="ROTATION",
    )

    @classmethod
    def poll(cls, context):
        return context.mode == "OBJECT"

    def execute(self, context):
        if self.target == "SELECTED":
            objects = [obj for obj in context.selected_objects if obj.type == "MESH"]
        else:
            objects = [
                obj for obj in context.view_layer.objects
                if obj.type == "MESH" and obj.visible_get()
            ]

        if not objects:
            self.report({"WARNING"}, "No target mesh objects found")
            return {"CANCELLED"}

        depsgraph = context.evaluated_depsgraph_get()
        backup_collection = (
            ensure_backup_collection(context.scene) if self.keep_backups else None
        )

        converted = 0
        fallback_count = 0
        skipped = 0

        # Blender stores ANGLE properties internally in radians.
        tolerance_degrees = max(0.05, min(10.0, self.normal_tolerance * 57.295779513))

        for obj in objects:
            source_obj, source_mesh = get_mesh_data(
                obj, depsgraph, self.use_evaluated
            )

            try:
                if source_mesh is None or not source_mesh.vertices:
                    skipped += 1
                    continue

                axes = find_orthogonal_axes_from_faces(
                    source_mesh, tolerance_degrees
                )

                if axes is None:
                    axes = covariance_axes(source_mesh)
                    fallback_count += 1

                if axes is None:
                    skipped += 1
                    continue

                bounds = oriented_bounds(source_mesh, axes)
                if bounds is None:
                    skipped += 1
                    continue

                minimums, maximums = bounds
                dimensions = [
                    maximums[i] - minimums[i] for i in range(3)
                ]

                if min(dimensions) <= 1e-9:
                    skipped += 1
                    continue

                if self.keep_backups:
                    backup = obj.copy()
                    backup.data = obj.data.copy()
                    backup.name = f"{obj.name}_ORIGINAL"
                    backup_collection.objects.link(backup)
                    backup.hide_set(True)
                    backup.hide_render = True

                old_mesh = obj.data
                materials = (
                    list(old_mesh.materials) if self.preserve_materials else []
                )

                new_mesh = make_oriented_box_mesh(
                    f"{obj.name}_Prism",
                    axes,
                    minimums,
                    maximums,
                )

                for material in materials:
                    new_mesh.materials.append(material)

                obj.data = new_mesh

                if self.use_evaluated:
                    obj.modifiers.clear()

                if old_mesh.users == 0:
                    bpy.data.meshes.remove(old_mesh)

                converted += 1

            finally:
                release_mesh(source_obj, self.use_evaluated)

        self.report(
            {"INFO"},
            (
                f"Converted {converted} object(s); "
                f"PCA fallback used on {fallback_count}; skipped {skipped}"
            ),
        )
        return {"FINISHED"}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class VIEW3D_PT_beam_prism_tools(Panel):
    bl_label = "Beam Prism Tools"
    bl_idname = "VIEW3D_PT_beam_prism_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "3D Print"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Create tightly fitted solid beam boxes")
        layout.operator(
            OBJECT_OT_beams_to_prisms.bl_idname,
            text="Convert Beams to Prisms",
            icon="MESH_CUBE",
        )


classes = (
    OBJECT_OT_beams_to_prisms,
    VIEW3D_PT_beam_prism_tools,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()