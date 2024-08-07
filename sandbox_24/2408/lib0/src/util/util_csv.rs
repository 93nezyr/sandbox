use anyhow::Result;
use std::fs;
use std::path::Path;

/// # About
/// 
/// 数値データが記載されたcsvを読み込みます.
/// 
/// ## Arguments
/// 
/// * `path` - ファイルパス
/// * `has_headers` - ヘッダーがあるかどうか
/// * `invalid_value` - 数値に変換できない場合に代入する値
/// 
/// ## Returns
/// 
/// * `anyhow::Result<Vec<Vec<f32>>>` - 読み込んだデータ
/// 
/// ## Examples
/// 
/// ```rust
/// use *::read_numerical_data_csv;
/// let path = "data.csv";
/// let data = read_numerical_data_csv(path, false, f32::MAX).unwrap();
/// println!("{:?}", data);
/// ```
/// 
/// ## Note
/// 
/// * ファイルが存在しない場合はエラーを返します
pub fn read_numerical_data_csv<P>(path: P, has_headers: bool, invalid_value: f32) -> Result<Vec<Vec<f32>>>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let file = fs::File::open(path)?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(has_headers)
        .flexible(true)
        .from_reader(file);
    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let record = record
            .iter()
            .map(|e| e.to_string().parse().unwrap())
            .collect::<Vec<f32>>();
        data.push(record);
    }
    let max_len = data.iter().map(|r| r.len()).max().unwrap();
    for record in data.iter_mut() {
        while record.len() < max_len {
            record.push(invalid_value);
        }
    }

    Ok(data)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_read_csv() {
        let path = "data.csv";
        let file = fs::File::create(path).unwrap();
        // let mut wtr = csv::Writer::from_writer(file);
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(false)
            .flexible(true)
            .from_writer(file);
        wtr.write_record(&["2", "3", "4"]).unwrap();
        wtr.write_record(&["1", "2"]).unwrap();
        wtr.flush().unwrap();

        let data = read_numerical_data_csv(path, false, f32::MAX).unwrap();
        println!("{:?}", data);
    }
}
