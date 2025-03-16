### streamlit page to take user_query and proposed_answer as input and get the final answer along with the summary of changes in the final output

import streamlit as st
from faff_qa import process_answer

def main():
    user_query = st.text_input('Enter your query')
    proposed_answer = st.text_area('Enter the proposed answer')
    if st.button('Process Answer'):
        result = process_answer(user_query, proposed_answer)
        ## heading
        st.header('Final Answer')
        st.write('Final Answer:', result["final_answer"]["final_answer"])

        ## summary of changes
        st.header('Summary of Changes')
        st.write('Adequacy Issues:', result["final_answer"]['adequacy_assessment'])
        st.write('Formatting Improvements:', result["final_answer"]["format_assessment"]["formatting_issues"])

if __name__ == '__main__':
    main()