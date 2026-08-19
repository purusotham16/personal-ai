import streamlit as st

from database import (
    get_conversations,
    search_conversations,
    get_messages,
    delete_conversation,
    clear_conversations
)


# ============================================================
# LOAD CONVERSATION
# ============================================================

def load_conversation(
    conversation_id
):

    messages = get_messages(
        conversation_id
    )

    st.session_state.messages = []

    for role, content in messages:

        st.session_state.messages.append(
            {
                "role": role,
                "content": content
            }
        )

    st.session_state.current_conversation = (
        conversation_id
    )


# ============================================================
# RENDER CHAT HISTORY
# ============================================================

def render_chat_history():

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        """
        <div style="
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 15px;
        ">
        💬 Chat History
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # SEARCH BOX
    # ========================================================

    search_text = st.sidebar.text_input(
        "🔍 Search conversations",
        placeholder="Search your chats...",
        key="chat_search_history"
    )


    # ========================================================
    # GET CONVERSATIONS
    # ========================================================

    if search_text.strip():

        conversations = search_conversations(
            search_text.strip()
        )

    else:

        conversations = get_conversations()


    # ========================================================
    # EMPTY HISTORY
    # ========================================================

    if not conversations:

        if search_text.strip():

            st.sidebar.caption(
                "🔍 No matching conversations found."
            )

        else:

            st.sidebar.caption(
                "No conversations yet."
            )

        return


    # ========================================================
    # DISPLAY CONVERSATIONS
    # ========================================================

    for conversation in conversations:

        conversation_id = conversation[0]

        title = conversation[1]


        # ====================================================
        # CONVERSATION BUTTONS
        # ====================================================

        col1, col2 = st.sidebar.columns(
            [5, 1]
        )


        # ====================================================
        # OPEN CONVERSATION
        # ====================================================

        with col1:

            if st.button(
                f"💬 {title}",
                key=f"chat_{conversation_id}",
                use_container_width=True
            ):

                load_conversation(
                    conversation_id
                )

                st.rerun()


        # ====================================================
        # DELETE CONVERSATION
        # ====================================================

        with col2:

            if st.button(
                "🗑️",
                key=f"delete_{conversation_id}"
            ):

                delete_conversation(
                    conversation_id
                )


                # --------------------------------------------
                # Clear currently opened conversation
                # --------------------------------------------

                if (
                    st.session_state.get(
                        "current_conversation"
                    )
                    == conversation_id
                ):

                    st.session_state.messages = []

                    st.session_state.current_conversation = None


                st.rerun()


    # ========================================================
    # HISTORY SETTINGS
    # ========================================================

    st.sidebar.markdown("")


    with st.sidebar.expander(
        "⚙️ History Settings"
    ):

        st.caption(
            f"{len(conversations)} conversation(s)"
        )

        st.markdown("")


        if st.button(
            "🗑️ Clear All History",
            use_container_width=True
        ):

            clear_conversations()

            st.session_state.messages = []

            st.session_state.current_conversation = None

            st.rerun()