pub mod multiple {
    use std::error::Error;
    use std::fs::File;
    use std::io::BufRead;

    #[derive(Debug, Clone)]
    pub struct DataPoint {
        pub x: Vec<f64>,
        pub y: f64,
    }

    fn sigmoid(number: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-number))
    }

    fn logistic_loss(estimated_y: f64, true_y: f64) -> f64 {
        -true_y * f64::ln(estimated_y) - (1.0 - true_y) * f64::ln(1.0 - estimated_y)
    }

    pub fn linear_regression(
        data: &[DataPoint],
        learning_rate: f64,
        iterations: u32,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut theta = vec![0.0; data[0].x.len()];
        let mut derivatives = vec![0.0; data[0].x.len()];

        let training_examples = data.len() as f64;

        let mut costs = vec![];

        for _ in 0..iterations {
            update_derivatives(data, training_examples, &theta, &mut derivatives);
            update_theta(learning_rate, &mut theta, &derivatives);
            costs.push(compute_cost(&data, &theta));
        }

        (theta, costs)
    }

    fn update_theta(learning_rate: f64, theta: &mut [f64], d_theta: &[f64]) {
        for i in 0..theta.len() {
            theta[i] = theta[i] - learning_rate * d_theta[i];
        }
    }

    fn update_derivatives(
        data: &[DataPoint],
        training_examples: f64,
        theta: &[f64],
        d_theta: &mut [f64],
    ) {
        for i in 0..theta.len() {
            d_theta[i] = data
                .iter()
                .map(|data_point| {
                    ((theta[0..]
                        .iter()
                        .zip(data_point.x.iter())
                        .map(|(&x, &y)| x * y)
                        .sum::<f64>())
                        - data_point.y)
                        * data_point.x[i]
                })
                .sum::<f64>()
                / training_examples;
        }
    }

    fn compute_cost(data: &[DataPoint], theta: &[f64]) -> f64 {
        data.iter()
            .map(|data_point| (estimate_y(theta, &data_point.x) - data_point.y))
            .map(|number| number * number)
            .sum::<f64>()
            / (2.0 * theta.len() as f64)
    }

    pub fn estimate_y(theta: &[f64], x: &[f64]) -> f64 {
        theta[..]
            .iter()
            .zip(x.iter())
            .map(|(&x, &y)| x * y)
            .sum::<f64>()
    }

    // load_data assumes the data has all columns but the last for input data, the last being the
    // target. The amount of features is thus the amount of columns minus 1, and during
    // preprocessing a feature is prepended being the value 1.0 for all data points, to estimate
    // the constant. Behaviour of the algorithm is not defined for incorrect data format. All
    // rows must have exactly one floating point value for all columns.
    pub fn load_data(file_name: &str) -> Result<Vec<DataPoint>, Box<dyn Error>> {
        let file = File::open(file_name)?;
        let lines = std::io::BufReader::new(file).lines();
        let mut data = Vec::new();

        for line in lines {
            if let Ok(line) = line {
                let values: Vec<f64> = line.split(',').map(|s| s.trim().parse().unwrap()).collect();
                let amount_features = values.len() - 1;
                let data_point = DataPoint {
                    x: std::iter::once(1.0)
                        .chain(values[0..amount_features].iter().cloned())
                        .collect(),
                    y: values[amount_features],
                };
                data.push(data_point);
            }
        }

        Ok(data)
    }

    pub fn mean_normalisers(
        data: &[DataPoint],
    ) -> (Vec<Box<dyn Fn(f64) -> f64>>, Vec<Box<dyn Fn(f64) -> f64>>) {
        let mut normalisers: Vec<Box<dyn Fn(f64) -> f64>> = vec![];
        let mut inverters: Vec<Box<dyn Fn(f64) -> f64>> = vec![];

        let amount_features = data[0].x.len();

        let constant = move |x: f64| -> f64 { x };
        normalisers.push(Box::new(constant));
        inverters.push(Box::new(constant));

        for feature in 1..amount_features {
            let maximum = data
                .iter()
                .map(|data_point| data_point.x[feature])
                .fold(f64::NEG_INFINITY, |max, x| if x > max { x } else { max });

            let minimum = data
                .iter()
                .map(|data_point| data_point.x[feature])
                .fold(f64::INFINITY, |min, x| if x < min { x } else { min });

            let mean = data
                .iter()
                .map(|data_point| data_point.x[feature])
                .sum::<f64>()
                / data.len() as f64;

            let normaliser = move |x: f64| -> f64 { (x - mean) / (maximum - minimum) };
            let inverter = move |x: f64| -> f64 { x * (maximum - minimum) + mean };

            normalisers.push(Box::new(normaliser));
            inverters.push(Box::new(inverter));
        }

        (normalisers, inverters)
    }

    pub fn normalise(
        data: &[DataPoint],
        normalisers: &[Box<dyn Fn(f64) -> f64>],
    ) -> Vec<DataPoint> {
        data.iter()
            .map(|data_point| DataPoint {
                x: data_point
                    .x
                    .iter()
                    .zip(normalisers.iter())
                    .map(|(&x, normaliser)| normaliser(x))
                    .collect(),
                y: data_point.y,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiple_linear_regression() {
        let data = multiple::load_data("data.csv").unwrap();

        let (normalisers, inverters) = multiple::mean_normalisers(&data);

        assert!(data[0].x.len() == normalisers.len());
        assert!(data[0].x.len() == inverters.len());

        let normalised_data = multiple::normalise(&data, &normalisers);

        let learning_rate = 1.0;
        let iterations = 5;

        let (estimated_theta, _) =
            multiple::linear_regression(&normalised_data, learning_rate, iterations);

        let given_x = vec![1.0, 15.0, 2.0];
        let normalised_x: Vec<f64> = given_x
            .clone()
            .iter()
            .zip(normalisers.iter())
            .map(|(&x, normaliser)| normaliser(x))
            .collect();

        let estimated_y = multiple::estimate_y(&estimated_theta, &normalised_x);

        let true_y = 240.0;

        let relative_difference = 1.0 - estimated_y / true_y;

        println!("normalised_x={:?}", normalised_x);
        println!("estimated_theta={:?}", estimated_theta);
        println!(
            "true_y={}, estimated_y={}; relative_difference={}",
            true_y, estimated_y, relative_difference
        );
        assert!(relative_difference < 0.01);
    }
}
