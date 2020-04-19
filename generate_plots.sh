

# Sample call to generate plots and results
# python3.5 plot_generation.py --learned_and_analytical True --n_actions 4 --chosen_objs 1 --number_of_samples 10 --error_type xy --combine_error False

# Generate plots for all combinations
number_of_samples=1
possible_n_actions="4 8"
possible_chosen_objs="1 4"
possible_error_types="xy xytheta all"
possible_combine_error="True False"
learned_and_analytical=False

for n_actions in $possible_n_actions; do
    for chosen_objs in $possible_chosen_objs; do
      for error_type in $possible_error_types; do
        for combine_error in $possible_combine_error; do
          echo ... Generating Resuts: n_actions $n_actions, chosen_objs $chosen_objs, error_type $error_type, combine_error $combine_error
          if [ $chosen_objs = 1 ]; then
            learned_and_analytical=True
          else
            learned_and_analytical=False
          fi
          python3.5 plot_generation.py --learned_and_analytical $learned_and_analytical --n_actions $n_actions --chosen_objs $chosen_objs --number_of_samples $number_of_samples --error_type $error_type --combine_error $combine_error
        done
      done
    done
  done
