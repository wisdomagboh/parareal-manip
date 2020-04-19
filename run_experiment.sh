
########################### Generate experimental data #######################


number_of_samples=1
num_cores=4

# Single object pushing using both learned and analytical models with 4 actions in a control sequence
python3.5 parareal_manipulation.py --learned_and_analytical True --n_actions 4 --chosen_objs 1 --num_cores $num_cores --number_of_samples $number_of_samples

# Multi-object pushing using only the learned model with 4 actions in a control sequence
python3.5 parareal_manipulation.py --learned_and_analytical False --n_actions 4 --chosen_objs 4 --num_cores $num_cores --number_of_samples $number_of_samples

# Single object pushing using both learned and analytical models with 8 actions in a control sequence
python3.5 parareal_manipulation.py --learned_and_analytical True --n_actions 8 --chosen_objs 1 --num_cores $num_cores --number_of_samples $number_of_samples

# Multi-object pushing using only the learned model with 8 actions in a control sequence
python3.5 parareal_manipulation.py --learned_and_analytical False --n_actions 8 --chosen_objs 4 --num_cores $num_cores --number_of_samples $number_of_samples
