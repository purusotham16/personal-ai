import streamlit as st
import streamlit.components.v1 as components


def copy_to_clipboard(text, button_id):

    escaped_text = (
        text
        .replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )

    components.html(
        f"""
        <div style="
            display: flex;
            justify-content: flex-start;
            margin-top: 4px;
            margin-bottom: 12px;
        ">

            <button
                id="{button_id}"
                onclick="copyText()"
                style="
                    background-color: transparent;
                    color: #9aa6b8;
                    border: 1px solid #343b49;
                    border-radius: 8px;
                    padding: 5px 12px;
                    cursor: pointer;
                    font-size: 13px;
                "
            >
                📋 Copy
            </button>

        </div>

        <script>

        function copyText() {{

            const text = `{escaped_text}`;

            navigator.clipboard.writeText(text)
                .then(function() {{

                    const button =
                        document.getElementById("{button_id}");

                    button.innerHTML = "✅ Copied!";

                    setTimeout(function() {{

                        button.innerHTML = "📋 Copy";

                    }}, 1500);

                }})
                .catch(function(error) {{

                    console.error(
                        "Copy failed:",
                        error
                    );

                }});

        }}

        </script>
        """,
        height=55
    )