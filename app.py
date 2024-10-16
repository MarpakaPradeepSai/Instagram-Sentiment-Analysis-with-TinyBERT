st.markdown(
    """
    <style>
    .main {
        background-color: #F0F2F6;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        border: none;
        border-radius: 25px;  /* Increased border radius for more curve */
        padding: 10px 24px;
        cursor: pointer;
        font-size: 16px;  /* Increased font size for better readability */
        transition: background-color 0.3s, transform 0.3s;  /* Smooth transition */
    }
    .stButton>button:hover {
        background-color: #0056b3;
        color: white; /* Keeps the text color white on hover */
        transform: scale(1.05);  /* Slightly enlarge button on hover */
    }
    .stButton>button:active {
        background-color: #0056b3;  /* Maintain the hover color when clicked */
        color: white;  /* Ensure text remains white on click */
        transform: scale(1); /* Prevents scaling down on click */
    }
    .prediction-box {
        border-radius: 25px;  /* Match the button's rounded shape */
        padding: 10px;  /* Padding for the box */
        text-align: center;
        font-size: 18px;  /* Font size for better readability */
    }
    .center-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)
