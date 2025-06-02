import bpy, os, sys, math, random
from pathlib import Path

# ——————————————————————————————————————————————————————————————————————
# 1) OPTIONAL AI INTEGRATION
#    Adjust 'project_path' to your local repo for ColorfulCanvasAI.
project_path = '/Users/komalshahid/Desktop/Bellevue University/DSC680/projects/project3-colorful-canvas'
if project_path not in sys.path:
    sys.path.append(project_path)

try:
    from src.milestone3.colorful_canvas_complete import ColorfulCanvasAI
    AI_AVAILABLE = True
    print("✅ ColorfulCanvasAI enabled")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI unavailable: {e}\n💡 Falling back to basic image processing")

# ——————————————————————————————————————————————————————————————————————
# AI Generator Class Definition
class AnamorphicBillboardGenerator:
    def __init__(self):
        if AI_AVAILABLE:
            self.ai = ColorfulCanvasAI()
            print("🧠 AI generator initialized")
        else:
            self.ai = None
            print("⚠️ Using fallback non-AI generator")
            
    def process_image_with_ai(self, img_path, effect_type, strength=1.0):
        if self.ai:
            print(f"🔄 Processing {img_path} with AI ({effect_type})")
            depth_map = self.ai.generate_depth_map(img_path)
            processed = None
            analysis = {}
            
            if effect_type == "shadow_box":
                processed = self.ai.create_shadow_box_effect(img_path, depth_map, strength)
            elif effect_type == "seoul_corner":
                processed = self.ai.create_seoul_corner_projection(img_path, depth_map)
            else:
                processed = img_path  # fallback to original
            
            # Save intermediate files
            base_dir = os.path.dirname(output_path) if 'output_path' in globals() else '/tmp'
            base_name = os.path.splitext(os.path.basename(output_path if 'output_path' in globals() else 'output'))[0]
            processed_path = os.path.join(base_dir, f"{base_name}_processed.png")
            depth_path = os.path.join(base_dir, f"{base_name}_depth.png")
            
            self.ai.save_image(processed, processed_path)
            self.ai.save_image(depth_map, depth_path)
            
            return processed_path, depth_path, analysis
        else:
            print("⚙️ Simple processing (no AI)")
            return img_path, None, {}
# ——————————————————————————————————————————————————————————————————————


# ——————————————————————————————————————————————————————————————————————
# HELPERS: scene cleanup, frame/screen, material creation
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for mesh in list(bpy.data.meshes): bpy.data.meshes.remove(mesh)
    for img in list(bpy.data.images):  bpy.data.images.remove(img)
    print("🧹 Scene cleared")


def create_billboard_frame(width=16, height=9, depth=0.5, frame_thickness=0.8):
    # Outer box
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,0))
    outer = bpy.context.active_object
    outer.name = "Frame_Outer"
    outer.scale = (width/2 + frame_thickness, height/2 + frame_thickness, depth)

    # Inner cutout
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0,0,0))
    inner = bpy.context.active_object
    inner.name = "Frame_Inner"
    inner.scale = (width/2, height/2, depth + 0.1)

    # Boolean difference
    bool_mod = outer.modifiers.new("FrameCut","BOOLEAN")
    bool_mod.operation = 'DIFFERENCE'
    bool_mod.object   = inner
    bpy.context.view_layer.objects.active = outer
    bpy.ops.object.modifier_apply(modifier=bool_mod.name)
    bpy.data.objects.remove(inner, do_unlink=True)

    # Create screen
    bpy.ops.mesh.primitive_plane_add(size=2, location=(0,0,-0.05))
    screen = bpy.context.active_object
    screen.name  = "Billboard_Screen"
    screen.scale = (width/2, height/2, 1)

    print(f"📐 Frame + screen created ({width}×{height})")
    return outer, screen


def create_material_with_image(name, image_path=None, emission_strength=2.0):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    emis = nodes.new("ShaderNodeEmission")
    emis.inputs['Strength'].default_value = emission_strength
    if image_path and os.path.exists(image_path):
        img = bpy.data.images.load(image_path)
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = img
        links.new(tex.outputs['Color'], emis.inputs['Color'])
    links.new(emis.outputs['Emission'], out.inputs['Surface'])
    return mat
# ——————————————————————————————————————————————————————————————————————


# ——————————————————————————————————————————————————————————————————————
# DISPLACEMENT: basic & AI-depth
def create_extruded_geometry_from_image(image_path, screen_obj, extrude_distance=3.0):
    if not os.path.exists(image_path):
        print(f"⚠️ Image not found: {image_path}")
        return screen_obj

    # ensure UVs
    bpy.context.view_layer.objects.active = screen_obj
    screen_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66)
    bpy.ops.object.mode_set(mode='OBJECT')

    # load image
    img = bpy.data.images.load(image_path)

    # material with true displacement (Cycles BOTH)
    mat = bpy.data.materials.new("DispMat_Basic"); mat.use_nodes=True
    mat.cycles.displacement_method = 'BOTH'
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    nodes.clear()
    texc = nodes.new('ShaderNodeTexCoord')
    imgn = nodes.new('ShaderNodeTexImage'); imgn.image = img
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position=0.0
    ramp.color_ramp.elements[1].position=1.0
    disp = nodes.new('ShaderNodeDisplacement')
    disp.inputs['Scale'].default_value = extrude_distance
    bsdf= nodes.new('ShaderNodeBsdfPrincipled'); bsdf.inputs['Roughness'].default_value=0.7
    out = nodes.new('ShaderNodeOutputMaterial')

    links.new(texc.outputs['UV'],    imgn.inputs['Vector'])
    links.new(imgn.outputs['Color'], ramp.inputs['Fac'])
    links.new(imgn.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(ramp.outputs['Color'], disp.inputs['Height'])
    links.new(bsdf.outputs['BSDF'],  out.inputs['Surface'])
    links.new(disp.outputs['Displacement'], out.inputs['Displacement'])

    screen_obj.data.materials.clear()
    screen_obj.data.materials.append(mat)

    # modifiers
    sub = screen_obj.modifiers.new('Subsurf','SUBSURF'); sub.levels=4; sub.render_levels=4
    tex = bpy.data.textures.new('DispTex_Basic','IMAGE'); tex.image=img
    dmod = screen_obj.modifiers.new('Displace','DISPLACE')
    dmod.texture = tex; dmod.texture_coords='UV'; dmod.uv_layer=screen_obj.data.uv_layers.active.name
    dmod.strength=extrude_distance; dmod.mid_level=0.5

    print("🔨 Extruded from basic image brightness")
    return screen_obj


def create_extruded_geometry_from_ai_depth(image_path, depth_path, screen_obj, extrude_distance=3.0):
    if not depth_path or not os.path.exists(depth_path):
        return create_extruded_geometry_from_image(image_path, screen_obj, extrude_distance)

    # ensure UVs
    bpy.context.view_layer.objects.active = screen_obj
    screen_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.uv.smart_project(angle_limit=66)
    bpy.ops.object.mode_set(mode='OBJECT')

    # load maps
    depth_img = bpy.data.images.load(depth_path)
    color_img = bpy.data.images.load(image_path) if os.path.exists(image_path) else None

    mat = bpy.data.materials.new("DispMat_AI"); mat.use_nodes=True
    mat.cycles.displacement_method='BOTH'
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    nodes.clear()
    texc = nodes.new('ShaderNodeTexCoord')
    dtex = nodes.new('ShaderNodeTexImage'); dtex.image=depth_img
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.color_ramp.elements[0].position=0.1
    ramp.color_ramp.elements[1].position=0.9
    disp = nodes.new('ShaderNodeDisplacement')
    disp.inputs['Scale'].default_value = extrude_distance*1.5
    bsdf= nodes.new('ShaderNodeBsdfPrincipled'); bsdf.inputs['Roughness'].default_value=0.7
    out = nodes.new('ShaderNodeOutputMaterial')
    links.new(texc.outputs['UV'],   dtex.inputs['Vector'])
    links.new(dtex.outputs['Color'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], disp.inputs['Height'])
    if color_img:
        ctex = nodes.new('ShaderNodeTexImage'); ctex.image=color_img
        links.new(texc.outputs['UV'],   ctex.inputs['Vector'])
        links.new(ctex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    links.new(disp.outputs['Displacement'], out.inputs['Displacement'])

    screen_obj.data.materials.clear()
    screen_obj.data.materials.append(mat)

    sub = screen_obj.modifiers.new('AI_Subsurf','SUBSURF'); sub.levels=5; sub.render_levels=5
    tex2 = bpy.data.textures.new('DispTex_AI','IMAGE'); tex2.image=depth_img
    dm = screen_obj.modifiers.new('AI_Displace','DISPLACE')
    dm.texture=tex2; dm.texture_coords='UV'; dm.uv_layer=screen_obj.data.uv_layers.active.name
    dm.strength=extrude_distance*1.5; dm.mid_level=0.3

    print("🔨 Extruded using AI depth map")
    return screen_obj
# ——————————————————————————————————————————————————————————————————————


# ——————————————————————————————————————————————————————————————————————
# FLOATING SHAPES & PARTICLES
def create_floating_elements(base_location=(0,0,0), count=10):
    objs=[]
    for i in range(count):
        t = i%4
        if   t==0: bpy.ops.mesh.primitive_cube_add()
        elif t==1: bpy.ops.mesh.primitive_uv_sphere_add()
        elif t==2: bpy.ops.mesh.primitive_cylinder_add()
        else:      bpy.ops.mesh.primitive_torus_add()
        obj = bpy.context.active_object
        obj.name = f"Floating_{i}"
        obj.location = (
            base_location[0] + random.uniform(-12,12),
            base_location[1] + random.uniform(-2,8),
            base_location[2] + random.uniform(2,8)
        )
        s = random.uniform(0.5,2.0)
        obj.scale = (s,s,s)
        obj.rotation_euler = (
            random.uniform(0,math.pi),
            random.uniform(0,math.pi),
            random.uniform(0,math.pi)
        )
        mat = create_material_with_image(f"Mat_Float_{i}")
        # random hue
        mat.node_tree.nodes['Emission'].inputs['Color'].default_value = (
            random.random(), random.random(), random.random(), 1
        )
        obj.data.materials.append(mat)
        objs.append(obj)
    print(f"✨ Spawned {count} floating elements")
    return objs


def create_particle_effects(emitter_obj, count=500):
    bpy.context.view_layer.objects.active = emitter_obj
    emitter_obj.select_set(True)
    bpy.ops.object.particle_system_add()
    psys = emitter_obj.particle_systems[-1].settings
    psys.count           = count
    psys.frame_start     = 1
    psys.frame_end       = 1
    psys.lifetime        = 120
    psys.emit_from       = 'FACE'
    psys.physics_type    = 'NEWTON'
    psys.normal_factor   = 2.0
    psys.factor_random   = 0.5
    psys.render_type     = 'OBJECT'
    # create tiny sphere proto
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.05)
    proto = bpy.context.active_object
    proto.name = "P_Proto"
    pmat = bpy.data.materials.new("P_Mat"); pmat.use_nodes=True
    nodes=pmat.node_tree.nodes; links=pmat.node_tree.links
    nodes.clear()
    out=nodes.new("ShaderNodeOutputMaterial")
    em=nodes.new("ShaderNodeEmission")
    em.inputs['Color'].default_value=(1,0.8,0.2,1)
    em.inputs['Strength'].default_value=5
    links.new(em.outputs['Emission'], out.inputs['Surface'])
    proto.data.materials.append(pmat)
    psys.instance_object = proto
    psys.use_rotation_instance = True
    print(f"🌟 Particle system added ({count} particles)")
    return emitter_obj
# ——————————————————————————————————————————————————————————————————————


# ——————————————————————————————————————————————————————————————————————
# CAMERA & LIGHTING
def setup_camera_for_anamorphic_view(billboard_location=(0,0,0)):
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    cam.name = "AnamorphCam"
    cam.location = (25, -15, 5)
    cam.rotation_euler = (math.radians(75), 0, math.radians(55))
    cam.data.lens = 35
    cam.data.clip_start = 0.1
    cam.data.clip_end = 1000
    bpy.context.scene.camera = cam
    print("📷 Anamorphic camera set")
    return cam


def setup_professional_lighting():
    # key
    bpy.ops.object.light_add(type='AREA', location=(10,-10,15))
    key = bpy.context.active_object; key.name="Key"; key.data.energy=500; key.data.size=10; key.data.color=(1,0.95,0.8)
    # fill
    bpy.ops.object.light_add(type='AREA', location=(-8,-5,8))
    fill= bpy.context.active_object; fill.name="Fill"; fill.data.energy=200; fill.data.size=15; fill.data.color=(0.8,0.9,1)
    # rim
    bpy.ops.object.light_add(type='SPOT', location=(0,10,12))
    rim = bpy.context.active_object; rim.name="Rim"; rim.data.energy=300; rim.data.spot_size=math.radians(45); rim.data.color=(0.9,0.7,1); rim.data.use_contact_shadow=True
    # ambient low
    W = bpy.context.scene.world or bpy.data.worlds.new("World")
    W.use_nodes=True
    bg = W.node_tree.nodes.get("Background")
    bg.inputs[0].default_value=(0.02,0.02,0.02,1); bg.inputs[1].default_value=0.15
    print("💡 Pro three-point lighting set")


def setup_seoul_lighting():
    # clear old lights
    for o in [o for o in bpy.data.objects if o.type=='LIGHT']:
        bpy.data.objects.remove(o, do_unlink=True)
    def _add(kind,loc,en,sz,col,**kw):
        bpy.ops.object.light_add(type=kind, location=loc)
        L=bpy.context.active_object; L.data.energy=en; L.data.color=col
        if kind=='AREA': L.data.size=sz
        if kind=='SPOT': L.data.spot_size=sz; L.data.use_contact_shadow=True
        for k,v in kw.items(): setattr(L.data,k,v)
    _add('AREA',(5,-8,3),700,15,(0.7,0.8,1.0))
    _add('AREA',(-6,-3,4),300,10,(0.8,0.6,1.0))
    _add('SPOT',(0,8,10),500,math.radians(60),(1,0.5,0.5))
    _add('POINT',(0,0,-5),100,0.1,(0.2,0.3,0.4))
    W=bpy.context.scene.world; W.use_nodes=True
    bg=W.node_tree.nodes.get("Background"); bg.inputs[0].default_value=(0.01,0.01,0.02,1); bg.inputs[1].default_value=0.02
    print("💡 Seoul LED–style lighting set")


def setup_render_settings(effect_type="shadow_box"):
    S=bpy.context.scene
    S.render.engine='CYCLES'; S.render.resolution_x=1920; S.render.resolution_y=1080; S.render.resolution_percentage=100
    S.cycles.samples=128; S.cycles.use_denoising=True
    vt=S.view_settings; vt.view_transform='Filmic'
    if effect_type=='seoul_corner':
        vt.look='Very High Contrast'; vt.exposure=0.5; S.cycles.samples=160
    else:
        vt.look='High Contrast'; vt.exposure=0.0
    print(f"🎞️ Render settings for {effect_type} done")
# ——————————————————————————————————————————————————————————————————————


# ——————————————————————————————————————————————————————————————————————
# STEP REGISTRY & STEP FUNCTIONS
# globals to share data between steps
image_path     = None
effect_type    = None
width, height  = None, None
extrude_dist   = None
float_count    = None
particle_count = None
output_path    = None

outer_frame = screen = generator = processed_img = depth_map = analysis = camera = None

def step_clear():            clear_scene()
def step_frame_placeholder(): globals().update(dict(outer_frame=outer_frame, screen=screen))
def step_frame():            globals().update(zip(('outer_frame','screen'), create_billboard_frame(width, height)))
def step_ai():               globals().update(generator=AnamorphicBillboardGenerator())
def step_ai_process():
    global processed_img, depth_map, analysis
    processed_img, depth_map, analysis = generator.process_image_with_ai(image_path, effect_type, strength=1.5)
def step_extrude():          globals().update(screen=create_extruded_geometry_from_ai_depth(processed_img, depth_map, screen, extrude_dist))
def step_floating():         create_floating_elements(screen.location, float_count)
def step_particles():        create_particle_effects(screen, particle_count)
def step_camera():           globals().update(camera=setup_camera_for_anamorphic_view(screen.location))
def step_lighting():
    (setup_seoul_lighting if effect_type=='seoul_corner' else setup_professional_lighting)()
def step_rendercfg():        setup_render_settings(effect_type)
def step_render():           bpy.context.scene.render.filepath = output_path; bpy.ops.render.render(write_still=True)

STEP_REGISTRY = [
    ("Clear Scene",          step_clear),
    ("Create Frame+Screen",  step_frame),
    ("Init AI Generator",    step_ai),
    ("AI–Process Image",     step_ai_process),
    ("Extrude Geometry",     step_extrude),
    ("Floating Elements",    step_floating),
    ("Particle Effects",     step_particles),
    ("Setup Camera",         step_camera),
    ("Setup Lighting",       step_lighting),
    ("Render Settings",      step_rendercfg),
    ("Final Render",         step_render),
]
# ——————————————————————————————————————————————————————————————————————


# ——————————————————————————————————————————————————————————————————————
# MAIN: parse args, run selected steps
if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image",         required=True)
    parser.add_argument("--effect",        default="shadow_box", choices=["shadow_box","seoul_corner"])
    parser.add_argument("--width",   type=float, default=16)
    parser.add_argument("--height",  type=float, default=9)
    parser.add_argument("--extrude", type=float, default=2.0)
    parser.add_argument("--float_count",    type=int, default=10)
    parser.add_argument("--particle_count", type=int, default=800)
    parser.add_argument("--start_at",       type=int, default=1)
    parser.add_argument("--stop_after",     type=int, default=99)
    parser.add_argument("--output",         default="/tmp/final_render.png")
    
    # Parse command line args
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = parser.parse_args(argv)
    
    # Set globals from args
    image_path = args.image
    effect_type = args.effect
    width, height = args.width, args.height
    extrude_dist = args.extrude
    float_count = args.float_count
    particle_count = args.particle_count
    output_path = args.output
    
    # Run pipeline steps
    start_index = max(0, args.start_at - 1)
    stop_index = min(len(STEP_REGISTRY), args.stop_after)
    
    for i, (step_name, step_func) in enumerate(STEP_REGISTRY[start_index:stop_index], start=start_index+1):
        print(f"[{i}/{len(STEP_REGISTRY)}] {step_name}...")
        try:
            step_func()
        except Exception as e:
            print(f"❌ Error in step {i} ({step_name}): {str(e)}")
            if i <= 3:  # If error in early steps, abort
                print("🛑 Critical error in early step, aborting pipeline")
                break
    
    print(f"✅ Pipeline completed. Output: {output_path}")