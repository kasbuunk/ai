use std::error::Error;
use std::fs::File;
use std::io::BufRead;

struct DataPoint {
    surface_m2: f64,
    price: f64,
}

fn univariate_linear_regression(
    data: Vec<DataPoint>,
    learning_rate: f64,
    iterations: u32,
) -> (f64, f64) {
    // Initialise parameter.

    let mut x0 = 0.0;
    let mut x1 = 0.0;

    // Update parameter.
    for _ in 0..iterations {
        // Calculate derivative w.r.t. surface_m2.
        let (derivative_x0, derivative_x1) = derivatives(&data, x0, x1);
        (x0, x1) = update_parameters(learning_rate, x0, x1, derivative_x0, derivative_x1);
    }

    (x0, x1)
}

fn update_parameters(
    learning_rate: f64,
    x0: f64,
    x1: f64,
    derivative_x0: f64,
    derivative_x1: f64,
) -> (f64, f64) {
    (
        x0 - learning_rate * derivative_x0,
        x1 - learning_rate * derivative_x1,
    )
}

fn derivatives(data: &Vec<DataPoint>, x0: f64, x1: f64) -> (f64, f64) {
    let training_examples = data.len();
    let derivative_sum_x0: f64 = data
        .iter()
        .map(|data_point| x0 + x1 * data_point.surface_m2 - data_point.price)
        .sum();

    let derivative_sum_x1: f64 = data
        .iter()
        .map(|data_point| {
            (x0 + x1 * data_point.surface_m2 - data_point.price) * data_point.surface_m2
        })
        .sum();

    let derivative_x0: f64 = derivative_sum_x0 / training_examples as f64;
    let derivative_x1: f64 = derivative_sum_x1 / training_examples as f64;

    (derivative_x0, derivative_x1)
}

fn estimate(x0: f64, x1: f64, surface_m2: f64) -> f64 {
    x0 + x1 * surface_m2
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
                    surface_m2: values[0],
                    price: values[1],
                };
                data.push(data_point);
            }
        }
    }

    Ok(data)
}

fn main() -> Result<(), Box<dyn Error>> {
    // load data from file
    let data = load_data("data.csv")?;

    let learning_rate = 0.000005;
    let iterations = 10_000_000;

    let (x0, x1) = univariate_linear_regression(data, learning_rate, iterations);

    let given_surface_m2 = 15.0;
    let estimated_price = estimate(x0, x1, given_surface_m2);
    println!(
        "Estimated price for a house with {} m2 surface area: EUR {}.",
        given_surface_m2, estimated_price
    );
    println!(
        "Estimation: (x0 + x1 * surface) = ({} + {} * {})",
        x0, x1, given_surface_m2
    );

    Ok(())
}
