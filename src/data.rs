use arrow_array::{
    Array, Date32Array, Date64Array, Float32Array, Float64Array, LargeStringArray, StringArray,
};
use arrow_schema::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::collections::HashSet;
use std::fmt;
use std::fs::File;
use std::io::Read;
use std::path::Path;

#[derive(Debug, PartialEq, Eq)]
pub enum DataError {
    EmptyAssets,
    EmptyAssetName {
        index: usize,
    },
    DuplicateAssetName(String),
    EmptyObservations,
    InsufficientObservations {
        actual: usize,
    },
    InvalidObservationRange {
        start: usize,
        end: usize,
        available: usize,
    },
    DateCountMismatch {
        dates: usize,
        observations: usize,
    },
    EmptyDate {
        row: usize,
    },
    DuplicateDate(String),
    RowWidthMismatch {
        row: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteReturn {
        row: usize,
        asset: usize,
    },
    NonFiniteCovariance {
        row: usize,
        col: usize,
    },
    PositionLengthMismatch {
        assets: usize,
        notionals: usize,
    },
    NonFiniteNotional {
        index: usize,
    },
    UnknownPositionAsset(String),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyAssets => write!(f, "returns must include at least one asset"),
            Self::EmptyAssetName { index } => {
                write!(f, "asset name at index {index} is empty")
            }
            Self::DuplicateAssetName(name) => write!(f, "duplicate asset name: {name}"),
            Self::EmptyObservations => write!(f, "returns must include at least one observation"),
            Self::InsufficientObservations { actual } => write!(
                f,
                "sample covariance requires at least 2 observations; got {actual}"
            ),
            Self::InvalidObservationRange {
                start,
                end,
                available,
            } => write!(
                f,
                "observation range {start}..{end} is invalid for {available} rows"
            ),
            Self::DateCountMismatch {
                dates,
                observations,
            } => write!(
                f,
                "date count ({dates}) does not match observation count ({observations})"
            ),
            Self::EmptyDate { row } => write!(f, "date at row {row} is empty"),
            Self::DuplicateDate(date) => write!(f, "duplicate observation date: {date}"),
            Self::RowWidthMismatch {
                row,
                expected,
                actual,
            } => write!(
                f,
                "return row {row} has {actual} values; expected {expected}"
            ),
            Self::NonFiniteReturn { row, asset } => {
                write!(f, "return at row {row}, asset index {asset} is not finite")
            }
            Self::NonFiniteCovariance { row, col } => {
                write!(f, "sample covariance at ({row}, {col}) is not finite")
            }
            Self::PositionLengthMismatch { assets, notionals } => write!(
                f,
                "position asset count ({assets}) does not match notional count ({notionals})"
            ),
            Self::NonFiniteNotional { index } => {
                write!(f, "position notional at index {index} is not finite")
            }
            Self::UnknownPositionAsset(name) => {
                write!(f, "position asset is missing from returns: {name}")
            }
        }
    }
}

impl std::error::Error for DataError {}

#[derive(Debug)]
pub enum PositionCsvError {
    Csv(csv::Error),
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    MissingHeader(&'static str),
    DuplicateHeader(&'static str),
    MissingValue {
        row: u64,
        column: &'static str,
    },
    InvalidNotional {
        row: u64,
        value: String,
    },
    NonFiniteNotional {
        row: u64,
        value: String,
    },
    Data(DataError),
}

impl fmt::Display for PositionCsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Csv(error) => write!(f, "invalid positions CSV: {error}"),
            Self::Io { path, source } => {
                write!(
                    f,
                    "could not read positions CSV `{}`: {source}",
                    path.display()
                )
            }
            Self::MissingHeader(name) => write!(f, "positions CSV is missing the `{name}` header"),
            Self::DuplicateHeader(name) => {
                write!(f, "positions CSV has duplicate `{name}` headers")
            }
            Self::MissingValue { row, column } => {
                write!(f, "positions CSV row {row} has an empty `{column}` value")
            }
            Self::InvalidNotional { row, value } => {
                write!(f, "invalid notional `{value}` on positions CSV row {row}")
            }
            Self::NonFiniteNotional { row, value } => {
                write!(
                    f,
                    "non-finite notional `{value}` on positions CSV row {row}"
                )
            }
            Self::Data(error) => write!(f, "invalid positions: {error}"),
        }
    }
}

impl std::error::Error for PositionCsvError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Csv(error) => Some(error),
            Self::Io { source, .. } => Some(source),
            Self::Data(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum ParquetReturnsError {
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    Parquet(parquet::errors::ParquetError),
    Arrow(arrow_schema::ArrowError),
    MissingDateColumn,
    DuplicateDateColumn,
    UnsupportedColumnType {
        column: String,
        expected: &'static str,
        actual: String,
    },
    NullValue {
        row: usize,
        column: String,
    },
    Data(DataError),
}

impl fmt::Display for ParquetReturnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(
                f,
                "could not read returns Parquet file `{}`: {source}",
                path.display()
            ),
            Self::Parquet(error) => write!(f, "invalid returns Parquet file: {error}"),
            Self::Arrow(error) => write!(f, "invalid Parquet record batch: {error}"),
            Self::MissingDateColumn => {
                write!(f, "returns Parquet file is missing the `date` column")
            }
            Self::DuplicateDateColumn => {
                write!(f, "returns Parquet file has duplicate `date` columns")
            }
            Self::UnsupportedColumnType {
                column,
                expected,
                actual,
            } => write!(
                f,
                "column `{column}` has type {actual}; expected {expected}"
            ),
            Self::NullValue { row, column } => {
                write!(
                    f,
                    "null value in returns Parquet row {row}, column `{column}`"
                )
            }
            Self::Data(error) => write!(f, "invalid returns data: {error}"),
        }
    }
}

impl std::error::Error for ParquetReturnsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parquet(error) => Some(error),
            Self::Arrow(error) => Some(error),
            Self::Data(error) => Some(error),
            _ => None,
        }
    }
}

/// Rectangular, date-aligned log returns. Dates and assets retain input order.
///
/// Missing or non-finite return values are rejected; this type does not
/// impute data or reorder observations.
#[derive(Debug)]
pub struct AlignedReturns {
    asset_names: Vec<String>,
    dates: Vec<String>,
    values: Vec<Vec<f64>>,
}

impl AlignedReturns {
    pub fn new(
        asset_names: Vec<String>,
        dates: Vec<String>,
        values: Vec<Vec<f64>>,
    ) -> Result<Self, DataError> {
        validate_asset_names(&asset_names, false)?;
        if values.is_empty() {
            return Err(DataError::EmptyObservations);
        }
        if dates.len() != values.len() {
            return Err(DataError::DateCountMismatch {
                dates: dates.len(),
                observations: values.len(),
            });
        }

        let mut seen_dates = HashSet::with_capacity(dates.len());
        for (row, date) in dates.iter().enumerate() {
            if date.trim().is_empty() {
                return Err(DataError::EmptyDate { row });
            }
            if !seen_dates.insert(date) {
                return Err(DataError::DuplicateDate(date.clone()));
            }
        }

        for (row, observation) in values.iter().enumerate() {
            if observation.len() != asset_names.len() {
                return Err(DataError::RowWidthMismatch {
                    row,
                    expected: asset_names.len(),
                    actual: observation.len(),
                });
            }
            for (asset, value) in observation.iter().enumerate() {
                if !value.is_finite() {
                    return Err(DataError::NonFiniteReturn { row, asset });
                }
            }
        }

        Ok(Self {
            asset_names,
            dates,
            values,
        })
    }

    /// Read a wide Parquet table with a string or native date `date` column and Float32/Float64 asset columns.
    pub fn from_parquet_path(path: impl AsRef<Path>) -> Result<Self, ParquetReturnsError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| ParquetReturnsError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).map_err(ParquetReturnsError::Parquet)?;
        let schema = builder.schema().clone();

        let date_indices: Vec<usize> = schema
            .fields()
            .iter()
            .enumerate()
            .filter_map(|(index, field)| (field.name() == "date").then_some(index))
            .collect();
        let date_index = match date_indices.as_slice() {
            [] => return Err(ParquetReturnsError::MissingDateColumn),
            [index] => *index,
            _ => return Err(ParquetReturnsError::DuplicateDateColumn),
        };

        let date_type = schema.field(date_index).data_type();
        if !matches!(
            date_type,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Date32 | DataType::Date64
        ) {
            return Err(ParquetReturnsError::UnsupportedColumnType {
                column: "date".into(),
                expected: "Utf8, LargeUtf8, Date32, or Date64",
                actual: date_type.to_string(),
            });
        }

        let mut asset_columns = Vec::new();
        let mut asset_names = Vec::new();
        for (index, field) in schema.fields().iter().enumerate() {
            if index == date_index {
                continue;
            }
            if !matches!(field.data_type(), DataType::Float32 | DataType::Float64) {
                return Err(ParquetReturnsError::UnsupportedColumnType {
                    column: field.name().clone(),
                    expected: "Float32 or Float64",
                    actual: field.data_type().to_string(),
                });
            }
            asset_columns.push(index);
            asset_names.push(field.name().clone());
        }
        validate_asset_names(&asset_names, false).map_err(ParquetReturnsError::Data)?;

        let mut reader = builder.build().map_err(ParquetReturnsError::Parquet)?;
        let mut dates = Vec::new();
        let mut values = Vec::new();
        let mut row_offset = 0usize;
        for batch in &mut reader {
            let batch = batch.map_err(ParquetReturnsError::Arrow)?;
            let date_array = batch.column(date_index);
            for row in 0..batch.num_rows() {
                if date_array.is_null(row) {
                    return Err(ParquetReturnsError::NullValue {
                        row: row_offset,
                        column: "date".into(),
                    });
                }
                let date = match date_type {
                    DataType::Utf8 => date_array
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .map(|array| array.value(row).to_owned()),
                    DataType::LargeUtf8 => date_array
                        .as_any()
                        .downcast_ref::<LargeStringArray>()
                        .map(|array| array.value(row).to_owned()),
                    DataType::Date32 => date_array
                        .as_any()
                        .downcast_ref::<Date32Array>()
                        .map(|array| date_from_epoch_days(i64::from(array.value(row)))),
                    DataType::Date64 => date_array
                        .as_any()
                        .downcast_ref::<Date64Array>()
                        .map(|array| date_from_epoch_days(array.value(row).div_euclid(86_400_000))),
                    _ => None,
                }
                .ok_or_else(|| ParquetReturnsError::UnsupportedColumnType {
                    column: "date".into(),
                    expected: "Utf8, LargeUtf8, Date32, or Date64",
                    actual: date_type.to_string(),
                })?;

                let mut observation = Vec::with_capacity(asset_columns.len());
                for &column in &asset_columns {
                    let array = batch.column(column);
                    let name = &asset_names[observation.len()];
                    if array.is_null(row) {
                        return Err(ParquetReturnsError::NullValue {
                            row: row_offset,
                            column: name.clone(),
                        });
                    }
                    let value = match array.data_type() {
                        DataType::Float32 => array
                            .as_any()
                            .downcast_ref::<Float32Array>()
                            .map(|array| array.value(row) as f64),
                        DataType::Float64 => array
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .map(|array| array.value(row)),
                        _ => None,
                    }
                    .ok_or_else(|| {
                        ParquetReturnsError::UnsupportedColumnType {
                            column: name.clone(),
                            expected: "Float32 or Float64",
                            actual: array.data_type().to_string(),
                        }
                    })?;
                    observation.push(value);
                }

                dates.push(date);
                values.push(observation);
                row_offset += 1;
            }
        }

        Self::new(asset_names, dates, values).map_err(ParquetReturnsError::Data)
    }

    pub fn asset_names(&self) -> &[String] {
        &self.asset_names
    }

    pub fn dates(&self) -> &[String] {
        &self.dates
    }

    pub fn values(&self) -> &[Vec<f64>] {
        &self.values
    }

    /// Estimate the unbiased sample covariance of log returns.
    ///
    /// The result is a flattened row-major `n_assets * n_assets` matrix.
    pub fn sample_covariance(&self) -> Result<Vec<f64>, DataError> {
        self.sample_covariance_for_range(0, self.values.len())
    }

    /// Estimate covariance using observations in `[start, end)`.
    pub fn sample_covariance_for_range(
        &self,
        start: usize,
        end: usize,
    ) -> Result<Vec<f64>, DataError> {
        if start >= end || end > self.values.len() {
            return Err(DataError::InvalidObservationRange {
                start,
                end,
                available: self.values.len(),
            });
        }
        let observations = &self.values[start..end];
        let n_observations = observations.len();
        if n_observations < 2 {
            return Err(DataError::InsufficientObservations {
                actual: n_observations,
            });
        }

        let n_assets = self.asset_names.len();
        let mut means = vec![0.0; n_assets];
        for observation in observations {
            for (asset, value) in observation.iter().enumerate() {
                means[asset] += value;
            }
        }
        for (asset, mean) in means.iter_mut().enumerate() {
            *mean /= n_observations as f64;
            if !mean.is_finite() {
                return Err(DataError::NonFiniteCovariance {
                    row: asset,
                    col: asset,
                });
            }
        }

        let mut covariance = vec![0.0; n_assets * n_assets];
        for observation in observations {
            for row in 0..n_assets {
                let row_deviation = observation[row] - means[row];
                for col in 0..=row {
                    covariance[row * n_assets + col] +=
                        row_deviation * (observation[col] - means[col]);
                }
            }
        }

        let denominator = (n_observations - 1) as f64;
        for row in 0..n_assets {
            for col in 0..=row {
                let value = covariance[row * n_assets + col] / denominator;
                if !value.is_finite() {
                    return Err(DataError::NonFiniteCovariance { row, col });
                }
                covariance[row * n_assets + col] = value;
                covariance[col * n_assets + row] = value;
            }
        }

        Ok(covariance)
    }
}

fn date_from_epoch_days(days: i64) -> String {
    let shifted_days = days + 719_468;
    let era = (if shifted_days >= 0 {
        shifted_days
    } else {
        shifted_days - 146_096
    }) / 146_097;
    let day_of_era = shifted_days - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let mut year = year_of_era + era * 400;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_part = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_part + 2) / 5 + 1;
    let month = month_part + if month_part < 10 { 3 } else { -9 };
    if month <= 2 {
        year += 1;
    }

    format!("{year:04}-{month:02}-{day:02}")
}

/// Position notionals keyed by asset name.
#[derive(Debug)]
pub struct Positions {
    asset_names: Vec<String>,
    notionals: Vec<f64>,
}

impl Positions {
    pub fn new(asset_names: Vec<String>, notionals: Vec<f64>) -> Result<Self, DataError> {
        validate_asset_names(&asset_names, true)?;
        if asset_names.len() != notionals.len() {
            return Err(DataError::PositionLengthMismatch {
                assets: asset_names.len(),
                notionals: notionals.len(),
            });
        }
        for (index, notional) in notionals.iter().enumerate() {
            if !notional.is_finite() {
                return Err(DataError::NonFiniteNotional { index });
            }
        }

        Ok(Self {
            asset_names,
            notionals,
        })
    }

    /// Read positions from CSV with `asset` and `notional` headers.
    pub fn from_csv_reader<R: Read>(reader: R) -> Result<Self, PositionCsvError> {
        let mut reader = csv::ReaderBuilder::new()
            .trim(csv::Trim::All)
            .from_reader(reader);
        let headers = reader.headers().map_err(PositionCsvError::Csv)?.clone();
        let asset_index = find_header(&headers, "asset")?;
        let notional_index = find_header(&headers, "notional")?;
        let mut asset_names = Vec::new();
        let mut notionals = Vec::new();

        for result in reader.records() {
            let record = result.map_err(PositionCsvError::Csv)?;
            let row = record
                .position()
                .map(|position| position.line())
                .unwrap_or(2);
            let asset = record
                .get(asset_index)
                .filter(|value| !value.trim().is_empty())
                .ok_or(PositionCsvError::MissingValue {
                    row,
                    column: "asset",
                })?;
            let notional_text = record
                .get(notional_index)
                .filter(|value| !value.trim().is_empty())
                .ok_or(PositionCsvError::MissingValue {
                    row,
                    column: "notional",
                })?;
            let notional =
                notional_text
                    .parse::<f64>()
                    .map_err(|_| PositionCsvError::InvalidNotional {
                        row,
                        value: notional_text.to_owned(),
                    })?;
            if !notional.is_finite() {
                return Err(PositionCsvError::NonFiniteNotional {
                    row,
                    value: notional_text.to_owned(),
                });
            }

            asset_names.push(asset.to_owned());
            notionals.push(notional);
        }

        Self::new(asset_names, notionals).map_err(PositionCsvError::Data)
    }

    pub fn from_csv_path(path: impl AsRef<Path>) -> Result<Self, PositionCsvError> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|source| PositionCsvError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::from_csv_reader(file)
    }

    /// Align positions to the returns asset order; assets without positions get zero.
    pub fn align_to(&self, returns: &AlignedReturns) -> Result<Vec<f64>, DataError> {
        let mut aligned = vec![0.0; returns.asset_names.len()];
        for (name, notional) in self.asset_names.iter().zip(&self.notionals) {
            let Some(index) = returns.asset_names.iter().position(|asset| asset == name) else {
                return Err(DataError::UnknownPositionAsset(name.clone()));
            };
            aligned[index] = *notional;
        }
        Ok(aligned)
    }
}

fn validate_asset_names(names: &[String], allow_empty: bool) -> Result<(), DataError> {
    if names.is_empty() && !allow_empty {
        return Err(DataError::EmptyAssets);
    }

    let mut seen = HashSet::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        if name.trim().is_empty() {
            return Err(DataError::EmptyAssetName { index });
        }
        if !seen.insert(name) {
            return Err(DataError::DuplicateAssetName(name.clone()));
        }
    }
    Ok(())
}

fn find_header(headers: &csv::StringRecord, name: &'static str) -> Result<usize, PositionCsvError> {
    let mut matching = headers
        .iter()
        .enumerate()
        .filter_map(|(index, header)| (header == name).then_some(index));
    let index = matching
        .next()
        .ok_or(PositionCsvError::MissingHeader(name))?;
    if matching.next().is_some() {
        return Err(PositionCsvError::DuplicateHeader(name));
    }
    Ok(index)
}
