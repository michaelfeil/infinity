//! Compatibility names for the shared model input types.
//!
//! New adapters should import `infinity_v3_core::input` directly.
#[allow(unused_imports)] // Compatibility exports may be unused by this binary.
pub use infinity_v3_core::input::{
    AudioFormat, AudioSource, ContentPart, ImageDetail, ImageSource, Message as DecisionMessage,
    MessageContent, MessageInput, MessageRole, ModelInput as SystemOneInput, VideoSource,
};

#[cfg(test)]
mod tests {
    use super::*;
    use utoipa::ToSchema;

    #[test]
    fn systemone_schema_uses_shared_type_names() {
        let (_, request) = super::super::systemone::SystemOneRequest::schema();
        let request = serde_json::to_value(request).unwrap();
        assert_eq!(
            request["properties"]["state"]["$ref"],
            "#/components/schemas/SystemOneState"
        );
        assert!(!request["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|key| key == "model"));
        let (_, state) = super::super::systemone::SystemOneState::schema();
        let state = serde_json::to_value(state).unwrap();
        let checked_in: serde_json::Value =
            serde_json::from_str(include_str!("../../../docs/openapi.json")).unwrap();
        assert_eq!(state, checked_in["components"]["schemas"]["SystemOneState"]);
        assert_eq!(
            request,
            checked_in["components"]["schemas"]["SystemOneRequest"]
        );
        let variants = state["oneOf"].as_array().unwrap();
        assert_eq!(variants.len(), 3);
        for (variant, expected) in variants.iter().zip(["string", "object", "array"]) {
            assert_eq!(variant["type"], expected);
        }
        let (_, messages) = MessageInput::schema();
        let messages = serde_json::to_value(messages).unwrap();
        assert_eq!(
            messages["properties"]["messages"]["items"]["$ref"],
            "#/components/schemas/Message"
        );
        assert_eq!(SystemOneInput::schema().0, "ModelInput");
        assert_eq!(DecisionMessage::schema().0, "Message");
    }
}
