pub mod regression {
    use std::error::Error;
    use std::fs::File;
    use std::io::BufRead;
    use std::rc::Rc;

    #[derive(Debug, Clone)]
    pub struct DataPoint {
        pub x: Vec<f64>,
        pub y: f64,
    }

    type RegressionModel = dyn Fn(&[f64], &[f64]) -> f64;

    type CostFn = dyn Fn(&[DataPoint], &[f64], f64) -> f64;

    fn inner_product(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f64>()
    }

    pub fn linear_regression(
        data: &[DataPoint],
        learning_rate: f64,
        regularisation_parameter: f64,
        iterations: u32,
    ) -> (Vec<f64>, Vec<f64>) {
        let model = Rc::new(linear_regression_model);
        let cost_function = Rc::new(compute_linear_cost);
        regression(
            data,
            learning_rate,
            regularisation_parameter,
            iterations,
            model,
            cost_function,
        )
    }

    pub fn logistic_regression(
        data: &[DataPoint],
        learning_rate: f64,
        regularisation_parameter: f64,
        iterations: u32,
    ) -> (Vec<f64>, Vec<f64>) {
        let model = Rc::new(logistic_regression_model);
        let cost_function = Rc::new(compute_logistic_cost);
        regression(
            data,
            learning_rate,
            regularisation_parameter,
            iterations,
            model,
            cost_function,
        )
    }

    fn compute_linear_cost(
        data: &[DataPoint],
        theta: &[f64],
        regularisation_parameter: f64,
    ) -> f64 {
        (data
            .iter()
            .map(|data_point| linear_loss(estimate_y(theta, &data_point.x), data_point.y))
            .sum::<f64>()
            + regularisation_parameter * theta.iter().sum::<f64>())
            / (2.0 * data.len() as f64)
    }

    fn compute_logistic_cost(
        data: &[DataPoint],
        theta: &[f64],
        regularisation_parameter: f64,
    ) -> f64 {
        (data
            .iter()
            .map(|data_point| logistic_loss(estimate_y(&data_point.x, theta), data_point.y))
            .sum::<f64>()
            + regularisation_parameter * theta.iter().sum::<f64>())
            / data.len() as f64
    }

    fn regression(
        data: &[DataPoint],
        learning_rate: f64,
        regularisation_parameter: f64,
        iterations: u32,
        model: Rc<RegressionModel>,
        cost_function: Rc<CostFn>,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut theta = vec![0.0; data[0].x.len()];
        let mut derivatives = vec![0.0; data[0].x.len()];

        let training_examples = data.len() as f64;

        let mut costs = vec![];

        for _ in 0..iterations {
            gradient_descent(
                data,
                training_examples,
                &theta,
                &mut derivatives,
                model.clone(),
                regularisation_parameter,
            );
            update_theta(learning_rate, &mut theta, &derivatives);
            costs.push(cost_function(&data, &theta, regularisation_parameter));
        }

        (theta, costs)
    }

    pub fn logistic_regression_model(feature_data: &[f64], theta: &[f64]) -> f64 {
        sigmoid(inner_product(feature_data, theta))
    }

    pub fn linear_regression_model(feature_data: &[f64], theta: &[f64]) -> f64 {
        inner_product(feature_data, theta)
    }

    fn linear_loss(estimated_y: f64, true_y: f64) -> f64 {
        (estimated_y - true_y).powi(2)
    }

    fn logistic_loss(estimated_y: f64, true_y: f64) -> f64 {
        -true_y * f64::ln(estimated_y) - (1.0 - true_y) * f64::ln(1.0 - estimated_y)
    }

    fn gradient_descent(
        data: &[DataPoint],
        training_examples: f64,
        theta: &[f64],
        d_theta: &mut [f64],
        model: Rc<RegressionModel>,
        regularisation_parameter: f64,
    ) {
        for (i, d_theta_element) in d_theta.iter_mut().enumerate() {
            *d_theta_element = (data
                .iter()
                .map(|data_point| (model(&data_point.x, theta) - data_point.y) * data_point.x[i])
                .sum::<f64>()
                + regularisation_parameter * theta[i])
                / training_examples;
        }
    }

    fn update_theta(learning_rate: f64, theta: &mut [f64], d_theta: &[f64]) {
        for i in 0..theta.len() {
            theta[i] = theta[i] - learning_rate * d_theta[i];
        }
    }

    pub fn estimate_y(theta: &[f64], x: &[f64]) -> f64 {
        theta[..]
            .iter()
            .zip(x.iter())
            .map(|(&x, &y)| x * y)
            .sum::<f64>()
    }

    fn sigmoid(number: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-number))
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
    use plotters::prelude::*;
    use plotters::style::RGBColor;

    #[test]
    fn multiple_linear_regression() {
        let data = regression::load_data("data/linear_regression.csv").unwrap();

        let (normalisers, inverters) = regression::mean_normalisers(&data);

        assert!(data[0].x.len() == normalisers.len());
        assert!(data[0].x.len() == inverters.len());

        let normalised_data = regression::normalise(&data, &normalisers);

        let learning_rate = 1.0;
        let regularisation_parameter = 0.1;
        let iterations = 5;

        let (estimated_theta, _) = regression::linear_regression(
            &normalised_data,
            learning_rate,
            regularisation_parameter,
            iterations,
        );

        let given_x = vec![1.0, 15.0, 2.0];
        let normalised_x: Vec<f64> = given_x
            .clone()
            .iter()
            .zip(normalisers.iter())
            .map(|(&x, normaliser)| normaliser(x))
            .collect();

        let estimated_y = regression::estimate_y(&estimated_theta, &normalised_x);

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
    // Function to plot a &[f64] data vector
    fn plot_data(data: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
        // Create a new file for the plot
        let root = BitMapBackend::new("test_cost_function.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;

        // Find the minimum and maximum values in the data vector
        let (_min_val, max_val) = data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &x| {
                (min.min(x), max.max(x))
            });

        // Define the chart area
        let mut chart = ChartBuilder::on(&root)
            .x_label_area_size(40)
            .y_label_area_size(80)
            .build_cartesian_2d(0.0..(data.len() as f64), 0.0..max_val)?;

        // Draw the data as a line plot
        chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

        chart.draw_series(LineSeries::new(
            data.iter().enumerate().map(|(x, y)| (x as f64, *y)),
            &RGBColor(255, 0, 0),
        ))?;

        Ok(())
    }
    #[test]
    fn logistic_regression() {
        let data = regression::load_data("data/logistic_regression1.csv").unwrap();

        let (normalisers, inverters) = regression::mean_normalisers(&data);

        assert!(data[0].x.len() == normalisers.len());
        assert!(data[0].x.len() == inverters.len());

        let normalised_data = regression::normalise(&data, &normalisers);

        let learning_rate = 1.0;
        let regularisation_parameter = 0.1;
        let iterations = 500;

        let (estimated_theta, costs) = regression::logistic_regression(
            &normalised_data,
            learning_rate,
            regularisation_parameter,
            iterations,
        );

        // Plot the data
        if let Err(err) = plot_data(&costs) {
            eprintln!("Error: {:?}", err);
        }

        let data_to_predict = vec![
            regression::DataPoint {
                x: vec![1.0, 90.0, 90.0],
                y: 1.0,
            },
            regression::DataPoint {
                x: vec![1.0, 50.0, 50.0],
                y: 0.0,
            },
        ];

        for data_point in &data_to_predict {
            let normalised_x: Vec<_> = data_point
                .x
                .iter()
                .zip(normalisers.iter())
                .map(|(&x, normaliser)| normaliser(x))
                .collect();

            let estimated_y =
                regression::logistic_regression_model(&normalised_x, &estimated_theta);

            assert!(estimated_y >= 0.0 && estimated_y <= 1.0);

            let relative_difference = 1.0 - estimated_y / data_point.y;

            println!("estimated_theta={:?}", estimated_theta);
            println!(
                "true_y={}, estimated_y={}; relative_difference={}",
                data_point.y, estimated_y, relative_difference
            );

            assert!(relative_difference < 0.01);
        }
    }
}
