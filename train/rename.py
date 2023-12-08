import os
import json
import model

if __name__ == "__main__":
    models_path = os.path.dirname(__file__) + "/../models/"

    last_num_matches_mean = 0

    # First read in the JSON contents of tmp
    with open(models_path + 'hyper_tmp.json') as infile:
        data = json.load(infile)
        last_num_matches_mean = data['last_num_matches_mean']

    base_path = model.model_filename_base(last_num_matches_mean)

    actor_path = models_path + "actor_" + base_path
    critic_path = models_path + "critic_" + base_path
    hp_path = models_path + "hyper_" + base_path + ".json"

    os.rename(models_path + 'actor_tmp', actor_path)
    os.rename(models_path + 'critic_tmp', critic_path)
    os.rename(models_path + 'hyper_tmp.json', hp_path)
