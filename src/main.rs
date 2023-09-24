use std::error::Error;

mod multiple {
    use std::error::Error;
    use std::fs::File;
    use std::io::BufRead;

    pub struct DataPoint {
        x: Vec<f64>,
        y: f64,
    }

    pub fn linear_regression(data: &[DataPoint], learning_rate: f64, iterations: u32) -> Vec<f64> {
        let mut theta = vec![0.0; data[0].x.len()];
        let mut derivatives = vec![0.0; data[0].x.len()];

        let training_examples = data.len() as f64;

        for _ in 0..iterations {
            update_derivatives(data, training_examples, &theta, &mut derivatives);
            update_theta(learning_rate, &mut theta, &derivatives);
        }

        theta
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

    pub fn estimate_y(theta: &[f64], x: &[f64]) -> f64 {
        theta[..]
            .iter()
            .zip(x.iter())
            .map(|(&x, &y)| x * y)
            .sum::<f64>()
    }

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
}

mod univariate {
    use std::error::Error;
    use std::fs::File;
    use std::io::BufRead;

    pub struct DataPoint {
        x: f64,
        y: f64,
    }

    pub fn linear_regression(
        data: Vec<DataPoint>,
        learning_rate: f64,
        iterations: u32,
    ) -> (f64, f64) {
        // Initialise parameter.
        let mut theta0 = 0.0;
        let mut theta1 = 0.0;

        // Update parameter.
        for _ in 0..iterations {
            // Calculate derivative w.r.t. theta0 and theta1.
            let (d_theta0, d_theta1) = derivatives(&data, theta0, theta1);
            (theta0, theta1) = update_parameters(learning_rate, theta0, theta1, d_theta0, d_theta1);
        }

        (theta0, theta1)
    }

    fn update_parameters(
        learning_rate: f64,
        theta0: f64,
        theta1: f64,
        d_theta0: f64,
        d_theta1: f64,
    ) -> (f64, f64) {
        (
            theta0 - learning_rate * d_theta0,
            theta1 - learning_rate * d_theta1,
        )
    }

    fn derivatives(data: &Vec<DataPoint>, theta0: f64, theta1: f64) -> (f64, f64) {
        let training_examples = data.len() as f64;

        (
            data.iter()
                .map(|data_point| theta0 + theta1 * data_point.x - data_point.y)
                .sum::<f64>()
                / training_examples,
            data.iter()
                .map(|data_point| (theta0 + theta1 * data_point.x - data_point.y) * data_point.x)
                .sum::<f64>()
                / training_examples,
        )
    }

    pub fn estimate_y(theta0: f64, theta1: f64, x: f64) -> f64 {
        theta0 + theta1 * x
    }

    pub fn load_data(file_name: &str) -> Result<Vec<DataPoint>, Box<dyn Error>> {
        let file = File::open(file_name)?;
        let lines = std::io::BufReader::new(file).lines();
        let mut data = Vec::new();

        for line in lines {
            if let Ok(line) = line {
                let values: Vec<f64> = line.split(',').map(|s| s.trim().parse().unwrap()).collect();
                if values.len() == 2 {
                    let data_point = DataPoint {
                        x: values[0],
                        y: values[1],
                    };
                    data.push(data_point);
                }
            }
        }

        Ok(data)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = multiple::load_data("data.csv")?;

    let learning_rate = 0.000005;
    let iterations = 10_000_000;

    let theta = multiple::linear_regression(&data, learning_rate, iterations);

    let given_x = vec![15.0, 2.0];
    let estimated_y = multiple::estimate_y(&theta, &given_x);
    println!("Given x = {:?}, estimated y is: {:?}", given_x, estimated_y);
    println!("Estimation: theta * x = {:?}", theta,);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiple_linear_regression() {
        let data = multiple::load_data("data.csv").unwrap();

        let learning_rate = 0.000005;
        let iterations = 1_000_000;

        let estimated_theta = multiple::linear_regression(&data, learning_rate, iterations);

        let given_x = vec![1.0, 15.0, 2.0];
        let true_theta = vec![10.0, 2.0, 100.0];
        let true_y: f64 = given_x
            .iter()
            .zip(true_theta.iter())
            .map(|(&x, &y)| x * y)
            .sum::<f64>();

        let estimated_y = multiple::estimate_y(&estimated_theta, &given_x);

        let relative_difference = 1.0 - estimated_y / true_y;

        println!(
            "true_theta={:?}, estimated_theta={:?}",
            true_theta, estimated_theta
        );
        println!(
            "true_y={}, estimated_y={}; relative_difference={}",
            true_y, estimated_y, relative_difference
        );
        assert!(relative_difference < 0.01);
    }
}
