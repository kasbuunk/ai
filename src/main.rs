use ai::multiple;
use std::error::Error;

use plotters::prelude::*;
use plotters::style::RGBColor;

// Function to plot a &[f64] data vector
fn plot_data(data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    // Create a new file for the plot
    let root = BitMapBackend::new("cost_function.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Find the minimum and maximum values in the data vector
    let (min_val, max_val) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
            (min.min(x), max.max(x))
        });

    // Define the chart area
    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(40)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0..(data.len() as f64), min_val..max_val)?;

    // Draw the data as a line plot
    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().enumerate().map(|(x, y)| (x as f64, *y)),
        &RGBColor(255, 0, 0),
    ))?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = multiple::load_data("data.csv")?;
    let (normalisers, _inverters) = multiple::mean_normalisers(&data);

    let normalised_data = multiple::normalise(&data, &normalisers);

    let learning_rate = 1.0;
    let iterations = 100;

    let (estimated_theta, costs) =
        multiple::linear_regression(&normalised_data, learning_rate, iterations);

    // Plot the data
    if let Err(err) = plot_data(&costs) {
        eprintln!("Error: {:?}", err);
    }

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
