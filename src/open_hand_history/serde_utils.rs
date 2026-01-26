use chrono::{DateTime, Utc};
use serde::{self, Deserialize, Deserializer, Serializer};

pub mod iso8601 {
    use super::*;

    pub fn serialize<S>(date: &Option<DateTime<Utc>>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match date {
            Some(date) => serializer.serialize_str(&date.to_rfc3339()),
            None => serializer.serialize_none(),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Option<DateTime<Utc>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s: Option<String> = Option::deserialize(deserializer)?;
        match s {
            Some(s) => {
                // Try parsing as RFC3339 first
                DateTime::parse_from_rfc3339(&s)
                    .or_else(|_| DateTime::parse_from_rfc3339(&format!("{s}Z")))
                    .map(|dt| Some(dt.with_timezone(&Utc)))
                    .map_err(serde::de::Error::custom)
            }
            None => Ok(None),
        }
    }
}

pub fn empty_string_is_none<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: serde::de::DeserializeOwned,
{
    #[derive(Deserialize, Debug)]
    #[serde(untagged)]
    enum Wrapper<T> {
        String(String),
        Result(T),
    }

    match Wrapper::deserialize(deserializer)? {
        Wrapper::String(s) if s.is_empty() => Ok(None),
        Wrapper::String(_) => Err(serde::de::Error::custom("expected empty string or vector")),
        Wrapper::Result(v) => Ok(Some(v)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize)]
    struct TestDate {
        #[serde(with = "iso8601")]
        date: Option<DateTime<Utc>>,
    }

    #[test]
    fn test_valid_dates() {
        let json = r#"{"date": "2023-11-14T12:34:56Z"}"#;
        let parsed: TestDate = serde_json::from_str(json).unwrap();
        assert!(parsed.date.is_some());

        let json = r#"{"date": "2023-11-14T12:34:56+00:00"}"#;
        let parsed: TestDate = serde_json::from_str(json).unwrap();
        assert!(parsed.date.is_some());

        let json = r#"{"date": "2023-11-14T12:34:56.123Z"}"#;
        let parsed: TestDate = serde_json::from_str(json).unwrap();
        assert!(parsed.date.is_some());
    }

    #[test]
    fn test_from_hand_history_example() {
        let json = r#"{"date": "2017-12-31T09:45:26Z"}"#;
        let parsed: TestDate = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.date.unwrap().to_rfc3339(),
            "2017-12-31T09:45:26+00:00"
        );
    }

    #[test]
    fn test_another_from_hh_example() {
        let json = r#"{"date": "2020-04-07T14:32:50"}"#;
        let parsed: TestDate = serde_json::from_str(json).unwrap();
        assert_eq!(
            parsed.date.unwrap().to_rfc3339(),
            "2020-04-07T14:32:50+00:00"
        );
    }

    #[test]
    fn test_invalid_dates() {
        // Invalid format
        let json = r#"{"date": "2023-11-14"}"#;
        assert!(serde_json::from_str::<TestDate>(json).is_err());

        // Invalid date
        let json = r#"{"date": "2023-13-14T12:34:56Z"}"#;
        assert!(serde_json::from_str::<TestDate>(json).is_err());

        // Invalid time
        let json = r#"{"date": "2023-11-14T25:34:56Z"}"#;
        assert!(serde_json::from_str::<TestDate>(json).is_err());
    }

    #[test]
    fn test_none() {
        let json = r#"{"date": null}"#;
        let parsed: TestDate = serde_json::from_str(json).unwrap();
        assert!(parsed.date.is_none());
    }

    #[test]
    fn test_roundtrip() {
        let original = TestDate {
            date: Some(Utc::now()),
        };

        let serialized = serde_json::to_string(&original).unwrap();
        let deserialized: TestDate = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            original.date.unwrap().to_rfc3339(),
            deserialized.date.unwrap().to_rfc3339()
        );
    }

    /// Test struct that uses empty_string_is_none deserializer
    #[derive(Debug, Deserialize)]
    struct TestVec {
        #[serde(deserialize_with = "super::empty_string_is_none")]
        values: Option<Vec<i32>>,
    }

    /// Verifies that an empty string deserializes to None.
    #[test]
    fn test_empty_string_is_none_empty() {
        let json = r#"{"values": ""}"#;
        let parsed: TestVec = serde_json::from_str(json).unwrap();
        assert!(parsed.values.is_none());
    }

    /// Verifies that a non-empty string causes a deserialization error
    /// since it's neither empty (which would be None) nor a valid array.
    #[test]
    fn test_empty_string_is_none_nonempty_string() {
        let json = r#"{"values": "not empty"}"#;
        let result = serde_json::from_str::<TestVec>(json);
        assert!(result.is_err(), "Non-empty string should cause an error");
    }

    /// Test that actual values deserialize correctly.
    #[test]
    fn test_empty_string_is_none_with_values() {
        let json = r#"{"values": [1, 2, 3]}"#;
        let parsed: TestVec = serde_json::from_str(json).unwrap();
        assert_eq!(parsed.values, Some(vec![1, 2, 3]));
    }
}
