bl_info = {
    "name": "Navisworks Import Cleanup",
    "author": "Your Name",
    "version": (0, 4, 0),
    "blender": (5, 0, 0),
    "location": "View3D > Sidebar > Navisworks",
    "description": (
        "Import FBX files, extract nested meshes, remove empty hierarchies, "
        "and move imported geometry to a specified center point"
    ),
    "category": "Import-Export",
}

import bpy

from mathutils import Vector
from bpy.props import BoolProperty, FloatVectorProperty, StringProperty
from bpy.types import Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ImportHelper


def collect_mesh_descendants(root_objects):
    """Find all unique mesh objects beneath the supplied root objects."""
    meshes = set()
    visited = set()
    stack = list(root_objects)

    while stack:
        obj = stack.pop()

        if obj in visited:
            continue

        visited.add(obj)

        if obj.type == "MESH":
            meshes.add(obj)

        stack.extend(obj.children)

    return list(meshes)


def get_world_bounds(objects):
    """Return world-space minimum and maximum bounds for the objects."""
    minimum = Vector((float("inf"), float("inf"), float("inf")))
    maximum = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)

            minimum.x = min(minimum.x, world_corner.x)
            minimum.y = min(minimum.y, world_corner.y)
            minimum.z = min(minimum.z, world_corner.z)

            maximum.x = max(maximum.x, world_corner.x)
            maximum.y = max(maximum.y, world_corner.y)
            maximum.z = max(maximum.z, world_corner.z)

    return minimum, maximum


def get_or_create_collection(name, scene):
    """Get or create a collection and link it to the current scene."""
    collection = bpy.data.collections.get(name)

    if collection is None:
        collection = bpy.data.collections.new(name)

    if scene.collection.children.get(collection.name) is None:
        scene.collection.children.link(collection)

    return collection


def move_object_to_collection(obj, target_collection):
    """Move an object exclusively into the target collection."""
    if target_collection.objects.get(obj.name) is None:
        target_collection.objects.link(obj)

    for collection in list(obj.users_collection):
        if collection != target_collection:
            collection.objects.unlink(obj)


def remove_empty_objects(objects):
    """Remove empty objects from a defined group of objects."""
    empties = [
        obj
        for obj in objects
        if obj.type == "EMPTY" and obj.name in bpy.data.objects
    ]

    empties.sort(
        key=lambda obj: len(obj.children_recursive),
        reverse=True,
    )

    removed_count = 0

    for empty in empties:
        if empty.name not in bpy.data.objects:
            continue

        if empty.children:
            continue

        bpy.data.objects.remove(empty, do_unlink=True)
        removed_count += 1

    return removed_count


def clean_imported_objects(context, source_objects):
    """
    Extract meshes, preserve world transforms, remove parenting,
    organize objects, move the assembly center to the requested
    coordinate, and remove empties.
    """
    settings = context.scene.navisworks_cleanup_settings
    source_objects = list(source_objects)

    meshes = collect_mesh_descendants(source_objects)

    if not meshes:
        raise RuntimeError(
            "No mesh objects were found in the imported FBX"
        )

    world_matrices = {
        obj: obj.matrix_world.copy()
        for obj in meshes
    }

    for obj in meshes:
        obj.parent = None
        obj.matrix_world = world_matrices[obj]

    target_collection = get_or_create_collection(
        settings.collection_name,
        context.scene,
    )

    for obj in meshes:
        move_object_to_collection(obj, target_collection)

    minimum, maximum = get_world_bounds(meshes)

    current_center = Vector((
        (minimum.x + maximum.x) * 0.5,
        (minimum.y + maximum.y) * 0.5,
        (minimum.z + maximum.z) * 0.5,
    ))

    target_center = Vector(settings.target_center)
    translation = target_center - current_center

    for obj in meshes:
        obj.matrix_world.translation += translation

    removed_empty_count = 0

    if settings.delete_empty_objects:
        removed_empty_count = remove_empty_objects(source_objects)

    bpy.ops.object.select_all(action="DESELECT")

    for obj in meshes:
        if obj.name in bpy.data.objects:
            obj.select_set(True)

    context.view_layer.objects.active = meshes[0]

    return meshes, removed_empty_count


class NavisworksCleanupSettings(PropertyGroup):
    collection_name: StringProperty(
        name="Collection Name",
        description="Collection used for cleaned mesh objects",
        default="YOUR_COLLECTION_NAME",
    )

    target_center: FloatVectorProperty(
        name="Target Center",
        description=(
            "World-space coordinate where the center of the imported "
            "assembly should be placed"
        ),
        default=(0.0, 0.0, 0.0),
        size=3,
        subtype="XYZ",
        unit="LENGTH",
    )

    delete_empty_objects: BoolProperty(
        name="Delete Empty Hierarchy",
        description=(
            "Delete imported empty objects after preserving "
            "the world transforms of their mesh children"
        ),
        default=True,
    )


class IMPORT_SCENE_OT_navisworks_fbx(Operator, ImportHelper):
    bl_idname = "import_scene.navisworks_fbx_cleanup"
    bl_label = "Import and Clean FBX"
    bl_description = (
        "Import an FBX file, remove its empty hierarchy, organize its meshes, "
        "and move the imported assembly to the target center"
    )
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".fbx"

    filter_glob: StringProperty(
        default="*.fbx",
        options={"HIDDEN"},
        maxlen=255,
    )

    def execute(self, context):
        existing_objects = set(bpy.data.objects)

        try:
            result = bpy.ops.import_scene.fbx(
                filepath=self.filepath,
            )
        except Exception as error:
            self.report(
                {"ERROR"},
                f"FBX import failed: {error}",
            )
            return {"CANCELLED"}

        if "FINISHED" not in result:
            self.report(
                {"ERROR"},
                "FBX import did not complete",
            )
            return {"CANCELLED"}

        imported_objects = [
            obj
            for obj in bpy.data.objects
            if obj not in existing_objects
        ]

        if not imported_objects:
            self.report(
                {"ERROR"},
                "The FBX importer did not create any objects",
            )
            return {"CANCELLED"}

        try:
            meshes, removed_empty_count = clean_imported_objects(
                context,
                imported_objects,
            )
        except RuntimeError as error:
            self.report(
                {"WARNING"},
                f"FBX imported, but cleanup failed: {error}",
            )
            return {"FINISHED"}

        self.report(
            {"INFO"},
            (
                f"Imported {len(imported_objects)} object(s), cleaned "
                f"{len(meshes)} mesh object(s), and removed "
                f"{removed_empty_count} empty object(s)"
            ),
        )

        return {"FINISHED"}


class VIEW3D_PT_navisworks_cleanup(Panel):
    bl_label = "Navisworks Cleanup"
    bl_idname = "VIEW3D_PT_navisworks_cleanup"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Navisworks"

    def draw(self, context):
        layout = self.layout
        settings = context.scene.navisworks_cleanup_settings

        layout.label(text="Import Settings")
        layout.prop(settings, "collection_name")
        layout.prop(settings, "target_center")
        layout.prop(settings, "delete_empty_objects")

        layout.separator()

        layout.operator(
            IMPORT_SCENE_OT_navisworks_fbx.bl_idname,
            text="Import and Clean FBX",
            icon="IMPORT",
        )


classes = (
    NavisworksCleanupSettings,
    IMPORT_SCENE_OT_navisworks_fbx,
    VIEW3D_PT_navisworks_cleanup,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.navisworks_cleanup_settings = bpy.props.PointerProperty(
        type=NavisworksCleanupSettings
    )


def unregister():
    if hasattr(
        bpy.types.Scene,
        "navisworks_cleanup_settings",
    ):
        del bpy.types.Scene.navisworks_cleanup_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()

