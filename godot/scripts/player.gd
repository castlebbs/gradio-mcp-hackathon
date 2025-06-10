extends CharacterBody3D


const SPEED = 3.0
const JUMP_VELOCITY = 4.5
@onready var camera_pivot: Node3D = $CameraPivot
@onready var player_mesh: Node3D = $PlayerMesh
#animation stuff
enum animation_state {IDLE, RUNNING, JUMPING}
var player_animation_state: animation_state = animation_state.IDLE
@export var animation_player: AnimationPlayer

func _physics_process(delta: float) -> void:
	# Add the gravity.
	if not is_on_floor():
		velocity += get_gravity() * delta

	# Handle jump.
	if Input.is_action_just_pressed("ui_accept") and is_on_floor():
		velocity.y = JUMP_VELOCITY

	# Get the input direction and handle the movement/deceleration.
	var input_dir := Input.get_vector("left", "right", "up", "down")
	var direction := (camera_pivot.basis * Vector3(input_dir.x, 0, input_dir.y)).normalized()
	if direction:
		velocity.x = direction.x * SPEED
		velocity.z = direction.z * SPEED
		#Rotate the playermesh
		player_mesh.basis = lerp(player_mesh.basis, Basis.looking_at(direction), 10.0 * delta)
		player_animation_state = animation_state.RUNNING
		
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)
		velocity.z = move_toward(velocity.z, 0, SPEED)
		player_animation_state = animation_state.IDLE

	move_and_slide()
	
	if not is_on_floor():
		player_animation_state = animation_state.JUMPING
		
	match player_animation_state:
		animation_state.IDLE:
			animation_player.speed_scale = 1.0
			animation_player.play("Armature|Idle")
		animation_state.RUNNING:
			animation_player.speed_scale = 1.5
			animation_player.play("Armature|Walk")
