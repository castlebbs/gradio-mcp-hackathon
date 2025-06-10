extends Node3D

const MOUSE_SENSITIVITY = 0.002
@onready var springarm : Node3D = $SpringArm3D

func _input(event: InputEvent) -> void:
	if event is InputEventMouseMotion:
		rotate_y(-event.relative.x * MOUSE_SENSITIVITY)
		springarm.rotation.x = clamp(springarm.rotation.x-event.relative.y * MOUSE_SENSITIVITY, -0.7, 0.7)
