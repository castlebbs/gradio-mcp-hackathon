extends Node
@onready var mcp_node = load("res://scripts/mcp.gd").new()
const number_of_assets = 2
signal assets_ready()

func _ready():

	# Connect to the signals
	mcp_node.output_ready.connect(_on_output_ready)
	mcp_node.command_failed.connect(_on_command_failed)
	await get_tree().process_frame


func extract_3d_assets_from_json(output_text: String = "res://assets/out_c.json", output_folder: String = "res://assets/") -> Array:
	"""
	Extracts 3D assets from JSON file and saves them as GLB files
	Returns array of extracted asset file paths
	"""
	
	var asset_paths = []
	
	# Parse JSON
	var json = JSON.new()
	var parse_result = json.parse(output_text)
	if parse_result != OK:
		print("Error: Failed to parse JSON")
		return asset_paths
	
	var data = json.data
	
	# Validate JSON structure
	if not data.has("assets") or not data.has("total_assets"):
		print("Error: Invalid JSON structure - missing 'assets' or 'total_assets'")
		return asset_paths
	
	var assets = data["assets"]
	var total_assets = data["total_assets"]
	
	print("Found ", total_assets, " assets to extract")
	
	# Create output directory if it doesn't exist
	if not DirAccess.dir_exists_absolute(output_folder):
		DirAccess.open("res://").make_dir_recursive(output_folder)
	
	var extracted_count = 0
	
	# Process each asset
	for asset in assets:
		if not asset.has("asset_id") or not asset.has("glb_data"):
			print("Warning: Asset missing required fields, skipping")
			continue
		
		var asset_id = asset["asset_id"]
		var glb_data_base64 = asset["glb_data"]
		
		# Decode base64 GLB data
		var glb_data = Marshalls.base64_to_raw(glb_data_base64)
		
		# Save GLB file
		var output_path = output_folder + asset_id + ".glb"
		var output_file = FileAccess.open(output_path, FileAccess.WRITE)
		
		if not output_file:
			print("Error: Could not create output file: ", output_path)
			continue
		
		output_file.store_buffer(glb_data)
		output_file.close()
		
		asset_paths.append(output_path)
		extracted_count += 1
		
	print("Successfully extracted ", extracted_count, " out of ", total_assets, " assets")
	return asset_paths

func place_assets_on_spawn_markers(asset_paths: Array) -> bool:
	if not is_inside_tree():
		print("Error: Node is not in the scene tree, deferring call")
		call_deferred("place_assets_on_spawn_markers", asset_paths)
		return false
	
	var scene_tree = get_tree()
	if not scene_tree:
		print("Error: Could not access scene tree")
		return false
	
	var main_scene = scene_tree.current_scene
	if not main_scene:
		print("Error: Could not access main scene")
		return false
	
	var placed_count = 0
	var max_spawns = min(asset_paths.size(), 10)  # Maximum 10 spawn points
	
	for i in range(max_spawns):
		var spawn_marker_name = "Spawn" + str(i + 1)
		var spawn_marker = main_scene.get_node_or_null(spawn_marker_name)
		
		if not spawn_marker:
			print("Warning: Spawn marker not found: ", spawn_marker_name)
			continue
		
		var asset_path = asset_paths[i]
		
		# Load the GLB file
		var gltf_document = GLTFDocument.new()
		var gltf_state = GLTFState.new()
		var error = gltf_document.append_from_file(asset_path, gltf_state)
		
		if error != OK:
			print("Error: Could not load GLB file: ", asset_path)
			continue
		
		# Generate the scene from GLTF
		var asset_scene = gltf_document.generate_scene(gltf_state)
		if not asset_scene:
			print("Error: Could not generate scene from GLB: ", asset_path)
			continue
		
		# Set the asset position to the spawn marker position
		asset_scene.global_transform.origin = spawn_marker.global_transform.origin
		#modify the scale of the asset
		var mesh_instance = find_mesh_instance(asset_scene)
		if mesh_instance:
			mesh_instance.scale = Vector3(1.5, 1.5, 1.5)  # Adjust scale as needed
		else:
			print("Warning: No MeshInstance3D found in asset scene: ", asset_path)

		# Add the asset to the main scene
		main_scene.add_child(asset_scene)
		
		placed_count += 1
		print("Placed asset at ", spawn_marker_name, ": ", asset_path)
	
	print("Successfully placed ", placed_count, " assets on spawn markers")
	assets_ready.emit()
	return placed_count > 0

func extract_and_place_assets(prompt: String) -> void:
	#this will call the MCP tools command to extract 3D assets
	#A signal will be emitted when the command is finished
	mcp_node.execute_mcptools_command(prompt, str(number_of_assets))  # Example command, adjust as needed


# Helper function to find MeshInstance3D recursively
func find_mesh_instance(node: Node) -> MeshInstance3D:
	if node is MeshInstance3D:
		return node as MeshInstance3D
	
	for child in node.get_children():
		var result = find_mesh_instance(child)
		if result:
			return result
	
	return null
#Function that catch the signal from MCP tools when the command is finished
func _on_output_ready(output_text: String, output_folder: String = "res://assets/"):
	print("Received AI output: ")
	# First extract the assets
	var extracted_assets = extract_3d_assets_from_json(output_text, output_folder)
	if extracted_assets.size() == 0:
		print("Error: Failed to extract assets")
		return false
	
	# Ensure we're in the scene tree before placing assets
	if not is_inside_tree():
		print("Deferring asset placement until node is in scene tree")
		call_deferred("place_assets_on_spawn_markers", extracted_assets)
		return true
	
	# Place the extracted assets on spawn markers
	return place_assets_on_spawn_markers(extracted_assets)

func _on_command_failed(error_message: String):
	print("AI asset generation failed failed: ", error_message)
	# Handle the error here
