import os
import sys
import gpt_2_simple as gpt2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

model_name = "124M"
file_name = "finetuning.txt"

#def splash():
#    print("0) New Game\n1) Load Game\n")
#    choice = get_num_options(2)
#
#    if choice == 1:
#        return "load"
#    else:
#        return "new"

def generate(prompt, session):
	return gpt2.generate(session,
		run_name='RUN',
		length=50,
		temperature=1,
		prefix=prompt,
		nsamples=1,
		batch_size=1,
		top_k=40,
		top_p=0.95,
		return_as_list=True)

def play():

#    console_print(
#        "Generating..."
#    )

	print("\nInitializing\n...This might take a few minutes\n")
	print("\n")
    
#	gpt2.download_gpt2(model_name=model_name)
	sess = gpt2.start_tf_sess()
	gpt2.load_gpt2(sess, run_name='RUN')

	with open("opening.txt", "r", encoding="utf-8") as file:
		starter = file.read()
	print(starter)

	while True:
		sys.stdin.flush()
		action = input("> ")
#		if action == "restart":
#			break

		if action == "quit":
			exit()

#		elif action == "help":
#			console_print(instructions())

#		elif action == "save":
#			if upload_story:
#				id = story_manager.story.save_to_storage()
#				console_print("Game saved.")
#				console_print(
#					"To load the game, type 'load' and enter the following ID: "
#					+ id
#				)
#			else:
#				console_print("Saving has been turned off. Cannot save.")

#		elif action == "load":
#			load_ID = input("What is the ID of the saved game?")
#			result = story_manager.story.load_from_storage(load_ID)
#			console_print("\nLoading Game...\n")
#			console_print(result)

#		elif action == "revert":
#
#			if len(story_manager.story.actions) == 0:
#				console_print("You can't go back any farther. ")
#				continue
#
#			story_manager.story.actions = story_manager.story.actions[:-1]
#			story_manager.story.results = story_manager.story.results[:-1]
#			console_print("Last action reverted. ")
#			if len(story_manager.story.results) > 0:
#				console_print(story_manager.story.results[-1])
#			else:
#				console_print(story_manager.story.story_start)
#			continue

		else:
			if action == "":
				action = ""
#				result = story_manager.act(action)
#				console_print(result)
				result = generate("", session)

#			elif action[0] == '"':
#				action = "You say " + action

			else:
				action = action.strip()
				action = action[0].lower() + action[1:]

#				if "You" not in action[:6] and "I" not in action[:6]:
#					action = "You " + action

				if action[-1] not in [".", "?", "!"]:
					action = action + "."

#				action = first_to_second_person(action)

#				action = "\n> " + action + "\n"

			result = "\n" + generate(action, session)

			console_print(result)


if __name__ == "__main__":
    play()
