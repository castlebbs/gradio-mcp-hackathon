extends Node3D

var ui_scene = preload("res://scenes/ui.tscn")
var ui_instance


func _ready() -> void:
	ui_instance = ui_scene.instantiate()
	add_child(ui_instance)
