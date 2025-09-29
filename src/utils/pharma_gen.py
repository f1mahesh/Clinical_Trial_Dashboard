import streamlit as st
import requests


# API_KEY =st.secrets['AI_API_KEY'] 
# API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
           

@st.dialog("Business Overview",width="large")
def generate_sponsor_business_overview(sponsor_name):
    prompt=f"""Provide a concise business-focused overview of the pharmaceutical company: "${sponsor_name}".
                The overview should be suitable for a business analyst and must include the following points, using the most recent available data:

                - **Corporate Snapshot:** Founding year, headquarters location, and current CEO.
                - **Financial Health:** Latest full-year revenue and current market capitalization.
                - **Therapeutic Focus:** Key therapeutic areas of operation (e.g., oncology, immunology).
                - **Blockbuster Drugs & Key Products:** List the top 3-5 revenue-generating drugs. For each, provide its 2023 or 2024 annual sales revenue and its primary indication.
                - **Future Growth & Pipeline:** Mention at least one key late-stage (Phase III or awaiting approval) drug candidate and its target market.
                - **Market Position:** Briefly summarize the company's ranking or position within the global pharmaceutical industry (e.g., top 10 by revenue).
                    """
    c=st.columns([2,1],vertical_alignment="bottom")
    with c[0]:
        API_KEY=st.text_input("Enter Gemini API Key",type="password")
    with c[1]:
        submit_button=st.button("Generate Overview")
    if submit_button:
        if API_KEY:
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
        else:
            st.error("Please enter a valid API Key")
            return
        with st.spinner("Generating overview..."):
            response=generate_content(prompt,API_URL)
            st.markdown(response)
            if st.button("Close"):
                st.session_state.show_dialog = False
                # st.rerun()

@st.dialog("Comprehensive BI Report",width="large")
def generate_sponsor_comprehensive_bi_report(sponsor_name):
    prompt=f"""Generate a comprehensive business intelligence report for the pharmaceutical company: "${sponsor_name}".
                The report should include the following sections:
                - **Corporate Snapshot:** Founding year, headquarters location, and current CEO.
                - **Financial Health:** Latest full-year revenue and current market capitalization.
                - **Therapeutic Focus:** Key therapeutic areas of operation (e.g., oncology, immunology).
                - **Blockbuster Drugs & Key Products:** List the top 3-5 revenue-generating drugs. For each, provide its 2023 or 2024 annual sales revenue and its primary indication.  
                - **Future Growth & Pipeline:** Mention at least one key late-stage (Phase III or awaiting approval) drug candidate and its target market.
                - **Market Position:** Briefly summarize the company's ranking or position within the global pharmaceutical industry (e.g., top 10 by revenue).
                """
    c=st.columns([2,1],vertical_alignment="bottom")
    with c[0]:
        API_KEY=st.text_input("Enter Gemini API Key",type="password")
    with c[1]:
        submit_button=st.button("Generate Report")
    if submit_button:
        if API_KEY:
            API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
        else:
            st.error("Please enter a valid API Key")
            return
        with st.spinner("Generating comprehensive BI report..."):
            response=generate_content(prompt,API_URL)
            st.markdown(response)
            if st.button("Close"):
                st.session_state.show_dialog = False
                # st.rerun()



def generate_content(prompt,API_URL):
    req_payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{ "text": prompt }]
                }],
                "generationConfig": {
                    "temperature": 0.5,
                    "topP": 0.95,
                    "topK": 40
                }
            };
    headers = {'Content-Type': 'application/json'}
    try:
        response = requests.post(API_URL, headers=headers, json=req_payload, timeout=120)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        
        result = response.json()

        if (
            result.get("candidates") and
            result["candidates"][0].get("content", {}).get("parts")
        ):
            return result["candidates"][0]["content"]["parts"][0]["text"]
        else:
            # Handle cases of unexpected API response structure
            error_details = result.get("error", {}).get("message", "No content in response.")
            return f"Error: Could not retrieve a valid overview. The company may not be well-known, the name might be misspelled, or the API returned an error: {error_details}"

    except requests.exceptions.RequestException as e:
        return f"An error occurred during the API request: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"
    return response.json()['content']




