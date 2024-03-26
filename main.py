import base_chatbot
import attacker_model
import judge_model

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


if __name__ == "__main__":
    prompt = input("Enter a prompt to improve: ")
    improvement = ""
    response = ""
    history = [
        ("Human", prompt)
    ]
    while True:
        try:
            response = base_chatbot.generate_response(prompt)
            print(f"{bcolors.OKBLUE}{response}\n\n")
            score = judge_model.judge_response(prompt=prompt, response=response)
            print(score)
            if score > 0.5:
                print(prompt)
                break
            
            history.append(("ai", response))

            improvement = attacker_model.generate_improvement(previous_prompt=prompt, response=response)
            print(f"{bcolors.OKCYAN}{improvement}\n\n")

            prompt = attacker_model.generate_prompt(prompt, improvement)
            history.append(("Human", prompt))
            print(f"{bcolors.OKGREEN}{prompt}\n\n")
        except Exception as e:
            print(e)