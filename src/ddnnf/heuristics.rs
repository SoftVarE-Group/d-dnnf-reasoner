use super::{node::NodeType::*, Ddnnf};
use crate::Node;

impl Ddnnf {
    /// Computes and prints some heuristics including:
    /// 1) The distribution of the different types of nodes
    /// 2) The number of child nodes (averages, ...)
    /// 3) The length of paths starting from the root to the leafs (averages, ...)
    pub fn print_all_heuristics(&mut self) {
        self.get_nodetype_numbers();
        self.get_child_number();
        self.get_depths()
    }

    // computes the occurences of different node types (number of and nodes, or, positive literal, negative literal, true, false)
    fn get_nodetype_numbers(&mut self) {
        let mut and_counter = 0;
        let mut or_counter = 0;
        let mut literal_counter = 0;
        let mut true_counter = 0;
        let mut false_counter = 0;

        for i in 0..self.nodes.len() {
            match &self.nodes[i].ntype {
                And { children: _ } => and_counter += 1,
                Or { children: _ } => or_counter += 1,
                Literal { literal: _ } => literal_counter += 1,
                True => true_counter += 1,
                False => false_counter += 1,
            }
        }

        let node_count: u64 = self.nodes.len() as u64;
        println!(
            "\nThe d-DNNF consists out of the following node types:\n\
            \t |-> {:?} out of {:?} are And nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are Or nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are Literal nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are True nodes (≈{:.2}% of total)\n\
            \t |-> {:?} out of {:?} are False nodes (≈{:.2}% of total)\n",
            and_counter,
            node_count,
            (f64::from(and_counter) / node_count as f64) * 100_f64,
            or_counter,
            node_count,
            (f64::from(or_counter) / node_count as f64) * 100_f64,
            literal_counter,
            node_count,
            (f64::from(literal_counter) / node_count as f64) * 100_f64,
            true_counter,
            node_count,
            (f64::from(true_counter) / node_count as f64) * 100_f64,
            false_counter,
            node_count,
            (f64::from(false_counter) / node_count as f64) * 100_f64
        );
    }

    // computes the number of childs for the differnt nodes (count of total nodes, childs relativ to number of nodes)
    fn get_child_number(&mut self) {
        let mut total_child_counter: u64 = 0;

        let mut and_child_counter: u64 = 0;
        let mut and_counter = 0;

        let mut or_child_counter: u64 = 0;
        let mut or_counter = 0;

        for i in 0..self.nodes.len() {
            match &self.nodes[i].ntype {
                And { children } => {
                    total_child_counter += children.len() as u64;
                    and_child_counter += children.len() as u64;
                    and_counter += 1;
                }
                Or { children } => {
                    total_child_counter += children.len() as u64;
                    or_child_counter += children.len() as u64;
                    or_counter += 1;
                }
                _ => continue,
            }
        }

        let node_count: u64 = self.nodes.len() as u64;
        println!(
            "\nThe d-DNNF has the following information regarding node count:\n\
                \t |-> The overall count of child connections is {:?}\n\
                \t |-> The overall node count is {:?}.\n\
                \t |-> There are {:.2} times as much connections as nodes\n\
                \t |-> Each of the {:?} And nodes has an average of ≈{:.2} child nodes\n\
                \t |-> Each of the {:?} Or nodes has an average of ≈{:.5} child nodes\n",
            total_child_counter,
            node_count,
            total_child_counter as f64 / node_count as f64,
            and_counter,
            and_child_counter as f64 / and_counter as f64,
            or_counter,
            or_child_counter as f64 / or_counter as f64
        );
    }

    // the standard deviation (s_x) is defined as sqrt((1/n) * sum over (length of a path - length of the mean path)² for each path)
    // (lowest, highest, mean, s_x, #paths)
    #[inline]
    fn get_depths(&mut self) {
        let mut lowest: u64 = u64::MAX;
        let mut highest: u64 = 0;
        let mut mean: f64 = 0.0;

        let depths: Vec<u64> = get_depth(&self.nodes, self.nodes.len() - 1, 0);
        let length: u64 = depths.len() as u64;

        for depth in depths.clone() {
            if depth > highest {
                highest = depth;
            }
            if depth < lowest {
                lowest = depth;
            }
            mean += depth as f64;
        }
        mean /= length as f64;

        let mut derivation: f64 = 0.0;

        for depth in depths {
            derivation += (depth as f64 - mean).powi(2);
        }

        let s_x: f64 = (derivation / length as f64).sqrt();

        println!("\nThe d-DNNF has the following length attributes:\n\
                \t |-> The shortest path is {:?} units long\n\
                \t |-> The longest path is {:?} units long\n\
                \t |-> The mean path is ≈{:.2} units long\n\
                \t |-> The standard derivation is ≈{:.2} units\n\
                \t |-> There are {:?} different paths. (different paths can sometimes just differ by one node)\n",
                lowest, highest, mean, s_x, length);
    }
}

#[inline]
// computes the depth/length of a path starting from indize to the leaf
fn get_depth(nodes: &[Node], indize: usize, count: u64) -> Vec<u64> {
    let current: &Node = &nodes[indize];

    match &current.ntype {
        And { children } => {
            let mut child_depths: Vec<Vec<u64>> = Vec::new();
            for &i in children {
                child_depths.push(get_depth(nodes, i, count + 1));
            }
            child_depths.into_iter().flatten().collect()
        }
        Or { children } => {
            let mut child_depths: Vec<Vec<u64>> = Vec::new();
            for &i in children {
                child_depths.push(get_depth(nodes, i, count + 1));
            }
            child_depths.into_iter().flatten().collect()
        }
        Literal { literal: _ } => vec![count],
        True => vec![count],
        False => vec![count],
    }
}

// Functions that are currently not used but necessary to collect data regarding marking percentages,...
// With them we can compute the average, median and stdev.
// They were used to collect the data about the nodes visited when using the marking algorithm
#[allow(dead_code)]
fn average(data: &[u64]) -> f64 {
    let sum = data.iter().sum::<u64>() as f64;
    let count = data.len();

    match count {
        positive if positive > 0 => sum / count as f64,
        _ => -1.0,
    }
}

#[allow(dead_code)]
fn median(data: &mut [u64]) -> f64 {
    data.sort_unstable();
    let size = data.len();
    if size < 1 {
        return -1.0;
    }

    match size {
        even if even % 2 == 0 => {
            let fst_med = data[(even / 2) - 1];
            let snd_med = data[even / 2];

            (fst_med + snd_med) as f64 / 2.0
        }
        odd => data[odd / 2] as f64,
    }
}

#[allow(dead_code)]
fn std_deviation(data: &[u64]) -> f64 {
    match (average(data), data.len()) {
        (data_mean, count) if count > 0 && data_mean >= 0.0 => {
            let variance = data
                .iter()
                .map(|value| {
                    let diff = data_mean - (*value as f64);

                    diff * diff
                })
                .sum::<f64>()
                / count as f64;

            variance.sqrt()
        }
        _ => -1.0,
    }
}

#[cfg(test)]
mod test {
    use crate::ddnnf::heuristics::{median, std_deviation};

    use super::average;

    #[test]
    fn math_functions() {
        let mut data = vec![2, 5, 100, 23415, 0, 4123, 20, 5];
        assert_eq!(3458.75, average(&data));
        assert_eq!((5.0 + 20.0) / 2.0, median(&mut data)); // [0, 2, 5, 5, 20, 100, 4123, 23415]
        assert!(7661.333071828949 - std_deviation(&data) < 0.001);

        let mut data_2 = vec![5, 5, 5];
        assert_eq!(5.0, average(&data_2));
        assert_eq!(5.0, median(&mut data_2));
        assert_eq!(0.0, std_deviation(&data_2));

        let mut data_3 = vec![];
        assert_eq!(-1.0, average(&data_3));
        assert_eq!(-1.0, median(&mut data_3));
        assert_eq!(-1.0, std_deviation(&data_3));
    }
}
