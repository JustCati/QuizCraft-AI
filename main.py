@cl.on_settings_update
async def setup_agent(settings):
    llm = OllamaLanguageModel(settings["Model"], settings["Temperature"]).get()
    cl.user_session.set("llm", llm)

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Gwen2.5 Model",
                values=["qwen2.5:7b", "qwen2.5:32b"],
                initial_index=0,
                initial="qwen2.5:32b",
            ),
            Switch(id="Streaming", label="Stream Tokens", initial=True),
            Slider(
                id="Temperature",
                label="Temperature",
                initial=0,
                min=0,
                max=1,
                step=0.1,
            ),
        ]
    ).send()

    await setup_agent(settings)
