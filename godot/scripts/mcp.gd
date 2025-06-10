extends Node

# Define the signal at the top of the class
signal output_ready(output_text: String)
signal command_failed(error_message: String)

var json_result = null
var thread: Thread
var output_array = []
var exit_code = -1



func execute_mcptools_command(prompt: String, assetNum: String):
	var script_path: String
	
	# Detect platform and use appropriate script
	if OS.get_name() == "Windows":
		script_path = ProjectSettings.globalize_path("res://mcp.bat")
	else:
		script_path = ProjectSettings.globalize_path("res://mcp.sh")
	
	var arguments = PackedStringArray([
		prompt,
		assetNum
	])
	
	# Create and start thread for non-blocking execution
	thread = Thread.new()
	thread.start(_execute_in_thread.bind(script_path, arguments))
	print("Command started in background thread using: ", script_path)

func _execute_in_thread(path: String, arguments: PackedStringArray):
	# Execute command and capture both stdout and stderr
	exit_code = OS.execute(path, arguments, output_array, true)
	
	# Call deferred to handle results on main thread
	call_deferred("_on_command_finished")

func _on_command_finished():
	print("Command finished with exit code: ", exit_code)
	
	# Wait for thread to complete and clean up
	if thread:
		thread.wait_to_finish()	
		thread = null
	
	# Process the results
	finalize_output()

func finalize_output():
	var output_text = ""
	
	# OS.execute appends all output to the array as strings
	if output_array.size() > 0:
		output_text = output_array[0]
		print("Command output length: ", output_text.length(), " characters")
	
	# If we captured stderr (when read_stderr was true), it might be mixed with stdout
	# or in some cases might be separate - this depends on the OS implementation
	if output_text.length() > 0:
		print("Saving JSON output...")
		save_output_to_file(output_text)
		output_ready.emit(output_text)
	else:
		print("No output received")
		if exit_code != 0:
			print("Command failed with exit code: ", exit_code)
			command_failed.emit("Command failed with exit code: " + str(exit_code))

func save_output_to_file(output: String):
	var timestamp = Time.get_datetime_string_from_system().replace(":", "-")
	var filename = "res://assets/mcp_output_" + timestamp + ".json"
	
	var file = FileAccess.open(filename, FileAccess.WRITE)
	if file:
		file.store_string(output)
		file.close()
		print("Output saved to: ", filename)
	else:
		print("Failed to save output to file")

func _exit_tree():
	if thread:
		thread.wait_to_finish()
		thread = null
