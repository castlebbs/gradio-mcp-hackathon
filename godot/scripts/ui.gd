extends Control

var asset_generator: Node

func _ready() -> void:
	# Create and add the asset generator to the scene tree
	asset_generator = Node.new()
	asset_generator.name = "AssetGenerator"
	asset_generator.set_script(preload("res://scripts/3Dgeneration.gd"))
	get_tree().current_scene.add_child(asset_generator)
	# Connect to the signals
	asset_generator.assets_ready.connect(_on_assets_ready)
	


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(_delta: float) -> void:
	pass

	


func _on_button_generate_pressed() -> void:

	var bio_text = $MarginContainer/ColorRect/MarginContainer/VBoxContainer/MarginContainer/VBoxContainer/BioTextEdit.text
	
	# Check if bio text is empty or only whitespace
	if bio_text.strip_edges().is_empty():
		# Show popup dialog
		$Alert.visible = true
		return

	$Loading.visible = true

	$MarginContainer/ColorRect/MarginContainer/VBoxContainer/MarginContainer/VBoxContainer/ButtonGenerate.disabled = true
	# Extract 3D assets from JSON
	if asset_generator:
		asset_generator.extract_and_place_assets(get_node("MarginContainer/ColorRect/MarginContainer/VBoxContainer/MarginContainer/VBoxContainer/BioTextEdit").text)

func _on_assets_ready():
	
	$Loading.visible = false
	
	$MarginContainer/ColorRect/MarginContainer/VBoxContainer/MarginContainer/VBoxContainer/ButtonGenerate.mouse_default_cursor_shape = Button.CURSOR_ARROW
	# Hide the UI scene when the button is clicked
	self.visible = false
	


func _on_close_btn_pressed() -> void:
	print("Close button pressed!") # Debug line
	if $Alert:
		$Alert.visible = false
	else:
		print("Alert node not found!")


func _on_close_button_pressed() -> void:
	print("Close button pressed!") # Debug line
	if $Alert:
		$Alert.visible = false
	else:
		print("Alert node not found!")
