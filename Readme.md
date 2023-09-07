# Negative Selection Algorithm Implementation

## Algorithm Details

- **Algorithm Implemented:** V detector

## Best Parameters

- **Expected coverage rate for non-self (c_0):** 0.999
- **Expected coverage rate for Self (c_1):** 0.9999
- **Self-radius:** 0.0081
- **Number of detectors:** 200
- **Best result:** 95.3% detection rate and 0.4% false alarm rate (Training for 30 seconds)

## Hyperparameter Tuning

### Tuning Parameters: C_0 and C_1

- The algorithm's termination is controlled by two hyperparameters, `c_0` and `c_1`.
- Setting `c_0` determines the self/non-self divide, and the goal is to maximize non-self detection.
- After experimentation, the following values were chosen:
  - `c_0`: 0.99
  - `c_1`: 0.9999
- When `c_0` was set to 0.95, it achieved a false alarm rate of 0.2% but limited the detection rate to 85.8% because only 64 out of 200 detectors were used.
- Increasing `c_0` to 0.999 resulted in a significant increase in the detection rate to 95.8% by utilizing all 200 detectors, with a slight increase in the false detection rate to 0.6%.
- Similar considerations were made for `c_1` to prevent early termination and maximize detector utilization.

### Self-radius (C_0: 0.99, C_1: 0.9999, Max detectors: 200)

- Experimentation with the self-radius parameter revealed a trade-off between detection rate and false alarm rate.
- Decreasing the self-radius increased the false alarm rate, e.g., at self-radius = 0.004, the false alarm rate was 4.8%, and the detection rate was 93.9%.
- To maximize both detection rate and minimize false alarm rate, the self-radius was found to be in the range of 0.0085 and 0.007.
- For example, at self-radius = 0.007 and 200 detectors, the detection rate was 94.8% with a 0.6% false alarm rate.
- Slight adjustments to the self-radius, such as 0.0081, resulted in a detection rate of 95.3% with a 0.4% false alarm rate but required a longer training time.

### Detector Number (C_0: 0.99, C_1: 0.9999, Self-radius: 0.0081)

- Experimentation with the number of detectors ranged from 170 to 220.
- At 170 detectors, a detection rate of 94.4% and a 0.4% false alarm rate were achieved.
- With 200 detectors, the detection rate increased to 95.3% with a 0.6% false alarm rate.
- However, increasing the number to 220 resulted in a higher false alarm rate (1.6%) due to excessive detectors.

# Graph colouring using Natural Algorithm (Bee Colony Algorithm Implementation)

## Algorithm Details

- **Algorithm Implemented:** Bee colony algorithm

## Best Result

- **Minimum Value:** 0.0
- **Minimum Point:** [-3.281580677104401e-09, 8.08915138459822e-09, -1.4676840570899771e-08]
- **Time Elapsed:** 15.4 seconds

## Best Parameters

- **Number of Employee Bees:** 40
- **Number of Onlooker Bees:** 40
- **Lambda:** 80
- **Fitness Function:** 1/computing function (aimed at minimizing the function and maximizing fitness)

## Hyperparameter Tuning

### Number of Employee Bees and Onlooker Bees (Lambda: 10, Running Time: 50s)

- Initially started with 20 employee bees and 20 onlooker bees, achieving a minimum of 0.005 ± 0.0005 for 10 runs.
- Increased the number of onlooker bees to 40, as they perform near neighbor search around the best solutions.
- Increasing onlookers to 40 resulted in a worse result of 0.01 ± 0.005, indicating the need for the same number of food sources as onlookers to have a variety of solutions.
- Further adjustments did not yield improvements.

### Lambda (Employee Bees: 40, Onlooker Bees: 40, Running Time: 50s)

- Lambda controls the abandonment of solutions in the bee colony.
- Increased lambda from 10 to 40, showing significant improvement with a solution of 6.23e-07.
- A low lambda means that good solutions are abandoned too quickly and can prevent onlookers from exploring near global minimum points.
- Increasing lambda to 100 guaranteed a 0.0 minimum value.
- Setting lambda to 500 took 120s for convergence due to infrequent abandonment of bad solutions.
- Found 80 to converge fastest (11s for 0).

## Experimentation

- Near-neighbor exploration of the solution space is controlled by a float change ranging between -1 and 1.
- Replaced constant range with an epoch-based learning schedule similar to one used in SVD.
- With this change, it converged to 0 at 20s compared to 11s before.
- The longer convergence time with decreased change could be due to improper exploration of the solution space.

## Fitness Function

- Initially, a fitness function compute f(i)/total fitness was used, but it had a shortcoming in minimization.
- The fitness function was changed to 1/compute f(i) to create a greater difference between different food sources.
- The new fitness function converges to 0.0 in 11s, while the previous fitness function takes 21s.
- In the code, it is possible to use the first fitness function by replacing `calc_fitness_population` with `calc_fitness_population_lectures`.
- The differentiation of probabilities encourages the exploration of better solutions, producing better near neighbors.
