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
            "#/components/schemas/ModelInput"
        );
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
