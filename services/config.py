from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
    openai_key: str
    pinecone_api_key: str

    model_config = SettingsConfigDict(env_file=".env")


config = AppConfig()
