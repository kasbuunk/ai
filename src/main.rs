use ai::multiple;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let data = multiple::load_data("data.csv")?;
    let (normalisers, _inverters) = multiple::mean_normalisers(&data);

    let normalised_data = multiple::normalise(&data, &normalisers);

    let learning_rate = 1.0;
    let iterations = 100;

    let (estimated_theta, costs) =
        multiple::linear_regression(&normalised_data, learning_rate, iterations);

    println!("costs: {:?}", costs);

    let given_x = vec![1.0, 15.0, 2.0];
    let normalised_x: Vec<f64> = given_x
        .clone()
        .iter()
        .zip(normalisers.iter())
        .map(|(&x, normaliser)| normaliser(x))
        .collect();

    let estimated_y = multiple::estimate_y(&estimated_theta, &normalised_x);

    println!("Given x = {:?}, estimated y is: {:?}", given_x, estimated_y);

    Ok(())
}
