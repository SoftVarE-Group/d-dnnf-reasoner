use super::{Fitness, Limit, Literal, Variable, DEFAULT_LIMIT, DEFAULT_SEED};
use nom::branch::alt;
use nom::bytes::complete::{tag, take_while};
use nom::character::complete::space1;
use nom::combinator::map_res;
use nom::error::{Error, ErrorKind};
use nom::multi::many1;
use nom::number::complete::recognize_float;
use nom::sequence::preceded;
use nom::{AsChar, Finish, IResult, Parser};
use nom_permutation::permutation_opt;
use std::ops::{Neg, RangeInclusive};
use std::path::PathBuf;
use std::str::FromStr;

/// Describes a query on a d-DNNF.
#[derive(Debug, PartialEq)]
pub enum Query {
    Count {
        assumptions: Vec<Literal>,
        /// A variable can also be deselected, we therefore parse it as a literal.
        variables: Vec<Literal>,
    },
    Core {
        assumptions: Vec<Literal>,
        /// A variable can also be deselected, we therefore parse it as a literal.
        variables: Vec<Literal>,
    },
    Sat {
        assumptions: Vec<Literal>,
        /// A variable can also be deselected, we therefore parse it as a literal.
        variables: Vec<Literal>,
    },
    Enumerate {
        limit: Limit,
        assumptions: Vec<Literal>,
    },
    Random {
        limit: Limit,
        seed: u64,
        assumptions: Vec<Literal>,
    },
    TWise {
        limit: Limit,
        fitness: Vec<Fitness>,
    },
    Atomic {
        cross: bool,
        candidates: Option<Vec<Variable>>,
        assumptions: Vec<Literal>,
    },
    SaveDdnnf {
        path: PathBuf,
    },
    SaveCnf {
        path: PathBuf,
    },
}

impl Query {
    /// Parses a query from a string.
    ///
    /// Returns `None` if unsuccessful.
    pub fn parse(input: &str) -> Result<Query, Error<&str>> {
        match query.parse(input.trim()).finish() {
            Ok((remaining, query)) => {
                // Remaining input indicates unsuccessful parsing.
                if !remaining.is_empty() {
                    return Err(Error::new(remaining, ErrorKind::Fail));
                }

                Ok(query)
            }
            Err(error) => Err(error),
        }
    }
}

// Each of the following functions is a parser for parts of a query string.
// The high level parsers are at the top, the lower level ones at the bottom.
// In general, the name of each combinator function describes the item it parses.
// The return type is always `IResult<&str, Type>` where `Type` is the type the function parses.

fn query(input: &str) -> IResult<&str, Query> {
    alt((
        count, core, sat, enumerate, random, t_wise, atomic, save_ddnnf, save_cnf,
    ))
    .parse(input)
}

fn count(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("count")(input)?;

    let (remaining, (variables, assumptions)) = permutation_opt((
        preceded(space1, variables_literals_full),
        preceded(space1, assumptions_full),
    ))
    .parse(remaining)?;

    Ok((
        remaining,
        Query::Count {
            variables: variables.unwrap_or_default(),
            assumptions: assumptions.unwrap_or_default(),
        },
    ))
}

fn core(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("core")(input)?;

    let (remaining, (variables, assumptions)) = permutation_opt((
        preceded(space1, variables_literals_full),
        preceded(space1, assumptions_full),
    ))
    .parse(remaining)?;

    Ok((
        remaining,
        Query::Core {
            variables: variables.unwrap_or_default(),
            assumptions: assumptions.unwrap_or_default(),
        },
    ))
}

fn sat(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("sat")(input)?;

    let (remaining, (variables, assumptions)) = permutation_opt((
        preceded(space1, variables_literals_full),
        preceded(space1, assumptions_full),
    ))
    .parse(remaining)?;

    Ok((
        remaining,
        Query::Sat {
            variables: variables.unwrap_or_default(),
            assumptions: assumptions.unwrap_or_default(),
        },
    ))
}

fn enumerate(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("enum")(input)?;

    let (remaining, (limit, assumptions)) =
        permutation_opt((preceded(space1, limit), preceded(space1, assumptions_full)))
            .parse(remaining)?;

    Ok((
        remaining,
        Query::Enumerate {
            limit: limit.unwrap_or(DEFAULT_LIMIT),
            assumptions: assumptions.unwrap_or_default(),
        },
    ))
}

fn random(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("random")(input)?;

    let (remaining, (limit, seed, assumptions)) = permutation_opt((
        preceded(space1, limit),
        preceded(space1, seed),
        preceded(space1, assumptions_full),
    ))
    .parse(remaining)?;

    Ok((
        remaining,
        Query::Random {
            limit: limit.unwrap_or(DEFAULT_LIMIT),
            seed: seed.unwrap_or(DEFAULT_SEED),
            assumptions: assumptions.unwrap_or_default(),
        },
    ))
}

fn t_wise(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("t-wise")(input)?;

    let (remaining, (limit, fitness)) =
        permutation_opt((preceded(space1, limit), preceded(space1, fitness_full)))
            .parse(remaining)?;

    Ok((
        remaining,
        Query::TWise {
            limit: limit.unwrap_or(DEFAULT_LIMIT),
            fitness: fitness.unwrap_or_default(),
        },
    ))
}

fn atomic(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("atomic")(input)?;
    let (remaining, cross) = alt((atomic_cross, success)).parse(remaining)?;

    let (remaining, (candidates, assumptions)) = permutation_opt((
        preceded(space1, variables_full),
        preceded(space1, assumptions_full),
    ))
    .parse(remaining)?;

    Ok((
        remaining,
        Query::Atomic {
            cross,
            candidates,
            assumptions: assumptions.unwrap_or_default(),
        },
    ))
}

fn save_ddnnf(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("save-ddnnf ")(input)?;
    let path = PathBuf::from(remaining.trim());
    Ok(("", Query::SaveDdnnf { path }))
}

fn save_cnf(input: &str) -> IResult<&str, Query> {
    let (remaining, _) = tag("save-cnf ")(input)?;
    let path = PathBuf::from(remaining.trim());
    Ok(("", Query::SaveCnf { path }))
}

/// Parses a list of literals preceded by `assumptions` or `a`.
fn assumptions_full(input: &str) -> IResult<&str, Vec<Literal>> {
    let (remaining, _) = alt((tag("assumptions"), tag("a"))).parse(input)?;
    literals.parse(remaining)
}

/// Parses a list of variables preceded by `variables` or `v`.
fn variables_full(input: &str) -> IResult<&str, Vec<Variable>> {
    let (remaining, _) = alt((tag("variables"), tag("v"))).parse(input)?;
    variables.parse(remaining)
}

/// Parses a list of literals preceded by `variables` or `v`.
fn variables_literals_full(input: &str) -> IResult<&str, Vec<Literal>> {
    let (remaining, _) = alt((tag("variables"), tag("v"))).parse(input)?;
    literals.parse(remaining)
}

/// Parses a list of literals.
fn literals(input: &str) -> IResult<&str, Vec<Literal>> {
    let (remaining, literals): (&str, Vec<Vec<Literal>>) =
        many1(preceded(space1, alt((range_signed, integer_signed_vec)))).parse(input)?;

    let literals = literals
        .into_iter()
        .flat_map(Vec::into_iter)
        .filter(|&literal| literal != 0)
        .collect();

    Ok((remaining, literals))
}

/// Parses a list of variables.
fn variables(input: &str) -> IResult<&str, Vec<Variable>> {
    let (remaining, variables): (&str, Vec<Vec<Variable>>) = many1(preceded(
        space1,
        alt((range_positive, integer_positive_vec)),
    ))
    .parse(input)?;

    let variables = variables
        .into_iter()
        .flat_map(Vec::into_iter)
        .filter(|&variable| variable != 0)
        .collect();

    Ok((remaining, variables))
}

/// Parses a list of floats preceded by `fitness` or `f`.
fn fitness_full(input: &str) -> IResult<&str, Vec<Fitness>> {
    let (remaining, _) = alt((tag("fitness"), tag("f"))).parse(input)?;
    many1(preceded(space1, float)).parse(remaining)
}

/// Returns successfully without consuming any input, returning the default implementation of the
/// given output type.
fn success<T: Default>(input: &str) -> IResult<&str, T> {
    Ok((input, T::default()))
}

/// Parses a limit such as `limit 1`.
fn limit(input: &str) -> IResult<&str, Limit> {
    let (remaining, _) = (alt((tag("limit"), tag("l"))), space1).parse(input)?;
    integer_positive.parse(remaining)
}

/// Parses a seed such as `seed 42`.
fn seed(input: &str) -> IResult<&str, u64> {
    let (remaining, _) = (alt((tag("seed"), tag("s"))), space1).parse(input)?;
    integer_positive.parse(remaining)
}

/// Parses a positive integer.
fn integer_positive<I: FromStr>(input: &str) -> IResult<&str, I> {
    map_res(take_while(char::is_dec_digit), |digits| I::from_str(digits)).parse(input)
}

/// Parses a positive integer, returning it in a `Vec` for combination with ranges.
fn integer_positive_vec<I: FromStr>(input: &str) -> IResult<&str, Vec<I>> {
    integer_positive(input).map(|(remaining, i): (&str, I)| (remaining, vec![i]))
}

/// Parses a negative integer.
fn integer_negative<I>(input: &str) -> IResult<&str, I>
where
    I: FromStr + Neg<Output = I>,
{
    preceded(tag("-"), integer_positive::<I>)
        .parse(input)
        .map(|(remaining, literal)| (remaining, -literal))
}

/// Parses a signed integer.
fn integer_signed<I>(input: &str) -> IResult<&str, I>
where
    I: FromStr + Neg<Output = I>,
{
    alt((integer_positive, integer_negative)).parse(input)
}

/// Parses a signed integer, returning it in a `Vec` for combination with ranges.
fn integer_signed_vec<I>(input: &str) -> IResult<&str, Vec<I>>
where
    I: FromStr + Neg<Output = I>,
{
    integer_signed(input).map(|(remaining, i): (&str, I)| (remaining, vec![i]))
}

/// Parses an inclusive range such as `1..42`.
fn range_positive<I>(input: &str) -> IResult<&str, Vec<I>>
where
    I: FromStr,
    RangeInclusive<I>: Iterator<Item = I>,
{
    let (remaining, (start, _, end)) =
        (integer_positive, tag(".."), integer_positive).parse(input)?;
    Ok((remaining, (start..=end).collect()))
}

/// Parses an inclusive range such as `-42..-1`.
fn range_signed<I>(input: &str) -> IResult<&str, Vec<I>>
where
    I: FromStr + Neg<Output = I>,
    RangeInclusive<I>: Iterator<Item = I>,
{
    let (remaining, (start, _, end)) = (
        alt((integer_positive, integer_negative)),
        tag(".."),
        alt((integer_positive, integer_negative)),
    )
        .parse(input)?;
    Ok((remaining, (start..=end).collect()))
}

/// Parses a float.
fn float(input: &str) -> IResult<&str, f64> {
    let (remaining, number) = recognize_float.parse(input)?;
    let parsed = f64::from_str(number).expect("Failed to parse float");
    Ok((remaining, parsed))
}

/// Parses the `-cross` string in case it is appended on `atomic`.
///
/// Indicates the presence of `-cross` via the return `bool`.
fn atomic_cross(input: &str) -> IResult<&str, bool> {
    let (remaining, _) = tag("-cross")(input)?;
    Ok((remaining, true))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn parse_integer() {
        assert_eq!(integer_positive::<u8>.parse("42").unwrap().1, 42);
    }

    #[test]
    fn parse_integer_negative() {
        assert_eq!(integer_negative::<i8>.parse("-42").unwrap().1, -42);
    }

    #[test]
    fn parse_range() {
        let result = range_positive::<u8>.parse("1..42").unwrap().1;
        let expected = (1..=42).collect::<Vec<u8>>();
        assert_eq!(result, expected);
    }

    #[test]
    fn parse_limit() {
        assert_eq!(limit.parse("l 42").unwrap().1, 42);
    }

    #[test]
    fn parse_seed() {
        assert_eq!(seed.parse("s 42").unwrap().1, 42);
    }

    #[test]
    fn parse_assumptions() {
        let input = "a 1 -2 3 -4";
        let expected = vec![1, -2, 3, -4];
        assert_eq!(assumptions_full.parse(input).unwrap().1, expected);
    }

    #[test]
    fn parse_assumptions_range() {
        let input = "a 1 -2 -4..-3";
        let expected = vec![1, -2, -4, -3];
        assert_eq!(assumptions_full.parse(input).unwrap().1, expected);
    }

    #[test]
    fn parse_variables() {
        let input = "v 1 2 3..5";
        let expected = vec![1, 2, 3, 4, 5];
        assert_eq!(variables_full.parse(input).unwrap().1, expected);
    }

    #[test]
    fn parse_variables_literals() {
        let input = "v 1 -2 3..5";
        let expected = vec![1, -2, 3, 4, 5];
        assert_eq!(variables_literals_full.parse(input).unwrap().1, expected);
    }

    #[test]
    fn parse_fitness() {
        let input = "f 1.0 4.2";
        let expected = vec![1f64, 4.2];
        assert_eq!(fitness_full.parse(input).unwrap().1, expected);
    }

    #[test]
    fn parse_core() {
        let input = "core a 1 2 3 v 4 5 6";
        let expected = Query::Core {
            assumptions: vec![1, 2, 3],
            variables: vec![4, 5, 6],
        };

        assert_eq!(Query::parse(input), Ok(expected));
    }
}
