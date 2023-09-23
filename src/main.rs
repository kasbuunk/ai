use std::error::Error;
use std::fs::File;
use std::io::BufRead;

struct DataPoint {
    x: f64,
    y: f64,
}

fn univariate_linear_regression(
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

fn estimate_y(theta0: f64, theta1: f64, x: f64) -> f64 {
    theta0 + theta1 * x
}

fn load_data(file_name: &str) -> Result<Vec<DataPoint>, Box<dyn Error>> {
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

fn main() -> Result<(), Box<dyn Error>> {
    let data = load_data("data.csv")?;

    let learning_rate = 0.000005;
    let iterations = 10_000_000;

    let (theta0, theta1) = univariate_linear_regression(data, learning_rate, iterations);

    let given_x = 15.0;
    let estimated_y = estimate_y(theta0, theta1, given_x);
    println!("Given x = {}, estimated y is: {}", given_x, estimated_y);
    println!(
        "Estimation: (theta0 + theta1 * x) = ({} + {} * {})",
        theta0, theta1, given_x
    );

    Ok(())
}
