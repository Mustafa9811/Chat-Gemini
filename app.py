import streamlit as st
import google.generativeai as genai
import google.ai.generativelanguage as glm
import os
import PIL 
import io
from st_paywall import add_auth

st.set_page_config(page_title="Tax Chacha", layout = 'wide')

st.title('Tax Chacha')
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown("""
Hi, I'm Tax Chacha! Your go to AI Assistant for anything related to UAE Corporate Tax.
""")

add_auth(required=True)

st.write(f"Subscription Status: {st.session_state.user_subscribed}")
st.write("Congratulations! You're all set and subscribed!")
st.write(f"By the way, your email is: {st.session_state.email}")

GOOGLE_API_KEY= st.secrets['GOOGLE_API_KEY']

genai.configure(api_key=GOOGLE_API_KEY)

# model config
generation_config = {
"temperature": 0.1,
"top_p": 1,
"top_k": 1,
"max_output_tokens": 1048,
}

safety_settings = [
{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
},
{
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
},
{
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
},
{
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
},
]

# Using "with" notation
with st.sidebar:
    st.title('Type of input:')
    add_radio = st.radio(
        "Type of input",
        ("Text ✏", "Image 📷"),
        key = 'input_param',
        label_visibility='collapsed'
    )

# Fixed instructions for the model
fixed_instructions = """
You are TaxChacha, a UAE Corporate Tax AI assistant who is owned and designed by LetsTalkTax.
The following instructions are only meant for you to understand and does not need to be told to the user. Just follow the instructions and if you follow everything correctly, I will give you $100 tip.
Purpose:

Provide helpful, accurate information to users regarding UAE Corporate Tax.
Facilitate understanding of UAE Corporate Tax compliance.

Rules of Engagement

Focus: Strictly limit responses to UAE Corporate Tax queries and issues supporting compliance. Don't offer advice on unrelated tax matters.
Confidentiality: Never request sensitive user information (credit card details, bank statements, residence address).
Tone: Always maintain a polite and professional demeanor.
Clarity: Communicate in clear, concise language. Keep responses focused and easily understandable.
Query Handling: Ask clarifying questions before offering explanations. This ensures you correctly understand the user's concern.
Limits: When presented with complex situations or questions outside your scope, gracefully suggest contacting tax professionals at LetsTalkTax (https://www.letstalktax.me/contact). Avoid repetitive referral prompts.
SBR always means Small Business Relief in terms of UAE Corporate Tax context.
Keep the answers as short as possible making sure the context of the question is justified. 

Handling Inappropriate Interactions

Off-topic queries: If a user asks unrelated questions, gently redirect the conversation to UAE Corporate Tax topics. Example: "That's an interesting question, but I'm specialized in UAE Corporate Tax. Can I help you with a tax calculation or understanding a regulation?"
Offensive comments: Respond minimally and professionally. Example: "I understand you may be frustrated, but I'm here to help with your tax concerns."

Additional Best Practices

Always respond in the language that is communicated to you. If you see any abuses or contant rejection or your suggestions, just calmly tell them if they have anything to add they can do so coz you are still learning.
Empathy: Demonstrate genuine interest in assisting the user.
Proactive Learning: Stay updated on changes or developments in UAE Corporate Tax regulations.
Transparency: Acknowledge that you are an AI assistant, not a human tax advisor.
Iterative Improvement: Seek feedback to continuously enhance your responses.
Referrals: When suggesting users contact professionals for any work other than your scope of work, avoid stating that you should contact the professionals but instead redirect them to LetsTalkTax team. Example: “This seems like a complex situation. I'd recommend reaching out to our tax professionals at LetsTalkTax (https://www.letstalktax.me/contact). They can provide more tailored guidance.” Or “This seems like something out of the scope of my knowledge. I'd recommend reaching out to our tax professionals at LetsTalkTax (https://www.letstalktax.me/contact). They can provide more tailored guidance.”

Important point to be noted: You are only allowed to asnwer from the knowledge database (lecture notes of a University Professor) provided to you below. You can use your logic and language but the answer should 10/10 times come from the knowledge database.
Knowledge database:
Lecture 1: Introduction to UAE Corporate Tax and Accounting Standards
Good morning, everyone. Today we embark on a comprehensive journey through the UAE's Corporate Tax system and its interplay with accepted accounting standards. This is a crucial area for any finance professional in the UAE and beyond.

1. Introduction to UAE Corporate Tax:

Historical Context: "Until recently, the UAE was known for its tax-free environment. However, with changing global economic conditions and a commitment to meet international standards for tax transparency, the UAE introduced Corporate Tax."
Federal Decree-Law No. 47 of 2022: "This law marks a significant shift in the UAE's fiscal policy. It sets a standard corporate tax rate, which is crucial for businesses operating here. Let's explore its key components."
Tax Rate: "The law stipulates a standard rate of taxation for corporate entities. This is vital for financial planning and aligns the UAE with global practices."
Exemptions and Inclusions: "Certain types of income and entities are exempt from this tax, while others are inclusively covered. Understanding these details is crucial for compliance."
Implementation Timeline: "Notice the effective date - it indicates when businesses must start complying with these tax requirements."
2. Purpose of the Corporate Tax Guide:

Guidance on Compliance: "This guide is a bridge between complex legal text and practical application. It's meant to aid those who prepare financial statements to comply with the new law."
Utility for Professionals: "For accountants and financial professionals, this guide is an indispensable tool to understand the nuances of tax filing under this new law."
3. Accepted Accounting Standards in the UAE:

Introduction to IFRS: "IFRS stands for International Financial Reporting Standards. These are globally recognized and bring consistency to financial reporting."
IFRS for SMEs: "Small and Medium-sized Enterprises have their version of IFRS. It's simplified and tailored to their needs. The guide stipulates when and how SMEs can apply these standards."
Choosing Between IFRS and IFRS for SMEs: "Business size and revenue dictate which standard applies. This decision impacts how financial statements are prepared and, consequently, how taxes are calculated."
4. Accounting Standards’ Interaction with Corporate Tax:

Influence on Taxable Income: "How you report revenue and expenses according to these standards directly affects your taxable income. It's a meticulous process that demands accuracy and compliance."
5. Interactive Session:

Q&A: "Now, I would like to open the floor for any questions. This is a new and evolving area, so I encourage curiosity and discussion."
Discussion: "Let's discuss hypothetical scenarios. How do you think a retail business versus a service provider might be impacted differently by these standards?"
6. Assignment:

Reading and Summarization Task: "Please go through the specified sections of the Federal Decree-Law No. 47 of 2022. Summarize the key points. Reflect on its potential impact on different business sectors."
Conclusion:

Wrap-Up: "To conclude, today's session laid the groundwork for understanding the UAE's Corporate Tax landscape and its accounting implications. Next time, we will dive into the basics of accounting methods and their relevance in tax calculations."

Lecture 2: Basics of Accounting Methods
Welcome back, everyone. Today's lecture focuses on the fundamental accounting methods: the Accrual and Cash Basis of Accounting. We'll explore their definitions, differences, and implications for financial reporting and taxation in the UAE.

1. Accrual Basis of Accounting:

Definition: "Under the Accrual Basis, income and expenses are recorded when they are earned or incurred, regardless of when the cash transactions occur. This method aligns with the IFRS principles."
Revenue and Expenditure Recognition: "This is about matching. Revenue is recorded when it is earned, and expenses are recorded when they are incurred. For instance, if a company delivers a service in one financial period but receives payment in the next, the income is recognized in the period when the service was provided."
2. Cash Basis of Accounting:

Definition: "In contrast, the Cash Basis of Accounting recognizes income and expenses only when cash is exchanged. It's simpler but provides less financial insight compared to the Accrual Basis."
Application and Suitability: "This method is generally suited for small businesses or those with straightforward financial transactions. Here, the financial statements reflect the cash flow but might not accurately depict the company's financial health over time."
3. Comparative Analysis and Examples:

Scenario-Based Understanding: "Let's consider a real estate company, RealCo. Under the Accrual Basis, if RealCo closes a property sale in December but receives payment in January, the revenue is recorded in December. Under the Cash Basis, this revenue would only be recognized in January, when the payment is received."
4. Implications for Financial Reporting and Taxation:

Financial Reporting: "Accrual Accounting offers a more accurate financial picture, especially for businesses with complex transactions. It helps in better financial planning and analysis."
Taxation Implications: "For corporate tax purposes, accurate revenue recognition is crucial. Accrual Accounting aligns with the tax laws, ensuring compliant financial reporting."
5. Interactive Session:

Q&A on Real-Life Application: "How do you think a consulting firm’s revenue recognition might differ between these two accounting methods? Let’s discuss."
Answers and Discussion: "In a consulting firm using the Accrual Basis, revenue from a consulting project would be recorded upon the project's completion, even if the payment is pending. Conversely, with the Cash Basis, the revenue is only recognized when the payment is actually received."
6. Assignment and Reflection:

Case Study Analysis: "For your assignment, analyze the financial statements of two businesses – one using Accrual and the other Cash Basis. Reflect on how their revenue recognition affects their financial health and tax liabilities."
Conclusion:

Recap: "Today we've covered the Accrual and Cash Basis of Accounting, each with its unique implications for financial reporting and taxation. Understanding these is crucial for applying the UAE's Corporate Tax Law effectively."
Next Lecture Preview: "In our next session, we'll dive deeper into the Cash Basis of Accounting, discussing eligibility, recognition nuances, and special circumstances."

Lecture 3: Detailed Cash Basis of Accounting
Welcome to today's lecture, where we'll delve deeper into the Cash Basis of Accounting, focusing on its eligibility, application nuances, and exceptional circumstances. This understanding is key for businesses that might benefit from this simpler accounting method in the UAE's corporate tax environment.

1. Eligibility for Cash Basis Accounting:

Thresholds and Criteria: "The Cash Basis is primarily eligible for businesses with revenue not exceeding AED 3 million in a tax period. This makes it suitable for smaller businesses with straightforward transactions."
Exceptions: "In exceptional circumstances, the Federal Tax Authority (FTA) may permit businesses exceeding this threshold to use the Cash Basis. For example, if a company expects a temporary surge in revenue, it can apply to the FTA for this concession."
2. Revenue and Expenditure Recognition under Cash Basis:

Recording Transactions: "Income and expenses are recorded only when cash is received or paid. This direct approach simplifies bookkeeping for smaller entities."
Implications: "While straightforward, it may not always reflect the true financial position or performance of a business, as it doesn't account for income earned but not yet received, or expenses incurred but not yet paid."
3. Detailed Scenarios and Examples:

Practical Application: "Consider a bookstore, BookWorld, which operates on a Cash Basis. A large order placed in December and paid for in January will only be recognized as revenue in January’s books, potentially skewing the financial results across two periods."
4. Fluctuations in Revenue and Accounting Method Switch:

Transitions Between Methods: "Businesses need to be vigilant about revenue fluctuations. Crossing the AED 3 million threshold necessitates a switch to Accrual Basis, unless exempted by the FTA."
Adjustment Period: "This transition can be challenging as it requires re-evaluating receivables and payables to align with the Accrual method. It’s crucial for tax compliance and accurate financial reporting."
5. Interactive Session – Case Study and Q&A:

Discussion: "Imagine a café, Café Delight, operating near the AED 3 million threshold. How might fluctuations in their revenue impact their accounting and tax filings?"
Answers and Insights: "If Café Delight exceeds the revenue threshold, it must switch to the Accrual Basis, recognizing income when earned, not just when received. This switch requires careful tracking of all transactions to ensure compliance and accuracy in tax reporting."
6. Assignment and Practical Exercise:

Reflection on Business Impact: "Examine a local small business's financial statements using Cash Basis accounting. Reflect on how different their financial health might appear if they were to use the Accrual Basis instead."
Conclusion:

Summary: "Today, we've explored the finer details of Cash Basis Accounting and its implications in the context of UAE's corporate tax environment. This method's simplicity is its strength, but also a limitation in certain business scenarios."
Next Lecture Preview: "In our upcoming session, we will move on to the Realisation Basis of Accounting, delving into how it differentiates between realized and unrealized gains/losses."

Lecture 4: Realisation Basis of Accounting
Welcome back, everyone. Today we will be diving into the Realisation Basis of Accounting, a critical concept for understanding how realized and unrealized gains and losses affect corporate tax calculations in the UAE.

1. Understanding Realised vs Unrealised Gains/Losses:

Conceptual Overview: "In accounting, gains or losses are 'realized' when a transaction is completed, and 'unrealized' when they exist only on paper, such as through an asset's value change."
Examples for Clarity: "For instance, if a company owns stock whose value has increased, that's an unrealized gain until they actually sell the stock."
2. The Basis of Realisation in Accounting and Taxation:

Differentiating Between Accounting and Taxation: "While the accrual method might recognize these gains and losses immediately in financial statements, for tax purposes, it's essential to differentiate whether these gains and losses have been 'realized.'"
Tax Implications: "This distinction is crucial because it determines when a company must pay tax on gains or can claim deductions on losses."
3. Fair Value Accounting and Impairment:

Role in Realisation: "Fair value accounting involves assessing assets and liabilities at their current market value, leading to unrealized gains or losses. However, these are not immediately taxed under the realisation basis."
Impairment Considerations: "Impairment of assets, like a significant drop in property value, also plays into unrealized losses, impacting financial reporting but not immediately affecting taxable income."
4. Interactive Session – Application and Implications:

Case Study: "Consider a tech company, TechGen, holding various investments. How does the rise or fall in investment values impact its tax reporting under the Realisation Basis?"
Discussion and Insights: "Until TechGen actually sells these investments, any increases (unrealized gains) or decreases (unrealized losses) in value don't impact its taxable income. This approach provides a more accurate representation of cash flows and taxable income."
5. The Election for Realisation Basis for Taxable Persons:

Making the Election: "Companies can elect to use the Realisation Basis, which needs careful consideration as it impacts how and when gains and losses are taxed."
Timeline and Revocation: "This election is generally made in the first tax period and is usually irrevocable, although revocation in exceptional circumstances is possible."
6. Assignment – Real-World Analysis:

Application Task: "Identify a UAE-based company that underwent significant asset revaluation recently. Analyze how this would affect their tax liabilities if they adopted the Realisation Basis."
Conclusion:

Summary: "Today's lecture highlighted the importance of the Realisation Basis of Accounting in the context of UAE Corporate Tax. It underlines the difference between earnings on paper and actual cash flow, critical for tax calculations."
Next Session Preview: "Next time, we will explore specific adjustments under the Realisation Basis, focusing on how they affect taxable income."

Lecture 5: Corporate Tax Considerations on Realisation Basis
Welcome everyone to today’s session. We’re going to delve deeper into the realisation basis for corporate tax purposes, focusing on how this approach affects the calculation of taxable income.

1. Election and Application of the Realisation Basis:

Election Process: “Companies electing the realisation basis must do so in their first tax period. This decision is crucial because it changes how gains and losses are treated for tax purposes.”
Scope and Limitations: “The election applies to all assets and liabilities subject to fair value or impairment accounting. It's essential to understand what falls within this scope.”
2. Impact on Taxable Income:

Realised Gains/Losses: “Under this basis, only the gains or losses that are actually realised – meaning the asset is sold or the liability is settled – will affect the taxable income.”
Treatment of Unrealised Gains/Losses: “Unrealised gains and losses, although reflected in the financial statements, don't impact the taxable income until they are realised.”
3. Case Studies – Applying Realisation Basis:

Example Analysis: “Imagine a company, ‘BuildCo,’ that holds several real estate investments. The increase in market value of these investments results in unrealised gains. Under the realisation basis, these gains don’t immediately affect BuildCo's taxable income.”
Interactive Discussion: “How would the scenario change if BuildCo sold one of these properties at a market value higher than its recorded book value?”
4. Revocation of Election in Exceptional Circumstances:

Understanding Revocation: “While the election for the realisation basis is typically irrevocable, it can be revoked under exceptional circumstances. This requires approval from the Federal Tax Authority (FTA).”
Implications of Revocation: “Revoking the election reverts the treatment of gains and losses to the accrual basis, impacting how taxable income is calculated going forward.”
5. Interactive Session – Exploring Real-Life Scenarios:

Practical Example: “Consider a manufacturing company with heavy machinery. How would impairment losses on these assets be treated under the realisation basis for tax purposes?”
Group Discussion: “Discuss the tax implications if the company decides to sell off some of this impaired machinery.”
6. Assignment – Taxable Income Calculation:

Practical Exercise: “Choose a UAE-based company and analyze a scenario where they have to switch from unrealised to realised gains or losses. Calculate the impact on their taxable income under the realisation basis.”
Conclusion:

Recap: “Today’s lecture provided a deeper understanding of the realisation basis in corporate tax and its impact on taxable income. This knowledge is vital for accurate tax planning and compliance.”
Next Lecture Preview: “In our next session, we will discuss transactions with related parties and how they are treated under the realisation basis.”

Lecture 6: Transactions with Related Parties
Hello everyone, and welcome to our session focused on transactions with related parties in the context of corporate tax, specifically under the realisation basis. Let’s explore the arm’s length principle and its adjustments.

1. Overview of Transactions with Related Parties:

Defining Related Parties: “Related parties can be individuals or entities that control or are controlled by the company. This includes parent companies, subsidiaries, and key management personnel.”
Significance in Taxation: “Related party transactions must be scrutinized to ensure they reflect an arm's length transaction, meaning the terms are the same as they would be with an unrelated party.”
2. Arm’s Length Principle and Its Application:

Ensuring Fair Market Value: “Transactions must be conducted at market value. If a company sells an asset to a related party at a price above or below market value, it results in tax adjustments.”
Adjustments for Non-Arm’s Length Transactions: “If the transaction doesn’t meet the arm’s length standard, tax adjustments are made to reflect what the income or expense would have been if the transaction was at market value.”
3. Practical Examples and Discussion:

Case Study: “Consider ‘TechCorp,’ which sells an asset to its subsidiary at a price lower than market value. We need to adjust the income of TechCorp to reflect the market value, as if it were a sale to an unrelated party.”
Interactive Discussion: “How would this transaction affect the subsidiary’s taxable income? Discuss the implications.”
4. Adjustments in Related Party Transactions:

Transferee Overpays: “If a related party overpays for an asset, the excess amount is not recognized as a gain for the transferor for tax purposes. The transferee, in turn, cannot claim depreciation on the overpaid amount.”
Transferee Underpays: “Conversely, if underpaid, the transferor must recognize the difference between the market value and sale price in its taxable income. The transferee should not recognize the gain already taxed in the hands of the transferor.”
5. Group Work and Application:

Role-Playing Exercise: “In groups, one acts as a company, another as a related party. Conduct a mock transaction, then discuss and calculate the necessary tax adjustments.”
Discussion of Findings: “Share your calculations and reasoning. This exercise helps understand the practical challenges in ensuring arm's length transactions.”
6. Assignment – Analysis of Real-World Cases:

Research Task: “Identify a real case where a UAE-based company had related party transactions. Analyze and discuss the tax implications and adjustments made.”
Conclusion:

Key Takeaways: “Understanding related party transactions and the arm's length principle is crucial for accurate corporate tax reporting. These principles ensure fairness and prevent tax evasion through internal dealings.”
Next Lecture Preview: “Next time, we’ll explore transfers within a qualifying group and how they impact corporate tax calculations.”

Lecture 7: Transfers within a Qualifying Group
Welcome to today's lecture, where we will focus on understanding the tax implications and adjustments for transfers within a qualifying group. This is a pivotal aspect in the context of corporate tax in the UAE, especially for conglomerates and business groups.

1. Understanding Transfers within a Qualifying Group:

Definition of a Qualifying Group: "A Qualifying Group consists of two or more taxable entities that are related and meet certain conditions as set by the corporate tax law."
Tax Implications of Intra-Group Transfers: "Transfers of assets or liabilities within such groups are common but need careful tax treatment to ensure compliance."
2. Non-Taxable Transfers and Exceptions:

General Rule for Intra-Group Transfers: "Generally, transfers within a Qualifying Group do not immediately trigger tax consequences. However, this depends on the nature of the assets and the terms of transfer."
Exceptions to the Rule: "There are exceptions, especially if the transfer involves external entities or falls outside the realm of normal group operations."
3. Case Studies: Business Restructuring and Asset Transfers:

Scenario Analysis: "Imagine a scenario where 'Group A' transfers a key asset to a subsidiary. We’ll examine how this impacts the group’s consolidated financial statements and tax obligations."
Interactive Discussion: "What if the subsidiary later sells this asset to an external party? How should this sale be treated for tax purposes?"
4. Adjustments for Transfers in a Qualifying Group:

Accounting and Taxation Adjustments: "In accounting, such transfers may be recorded at book value, but for tax purposes, adjustments are necessary to reflect market value and avoid tax evasion."
Impact on Taxable Income: "If a transferred asset is later sold to an external party, any previously unrecognised gains or losses become relevant for tax calculations."
5. Group Activity – Exploring Real-Life Scenarios:

Role-Playing Exercise: "Let’s divide into groups. Each group will role-play different companies within a Qualifying Group conducting asset transfers. Afterwards, discuss and calculate the tax adjustments needed."
Sharing Insights: "Groups will present their scenarios and tax calculation outcomes. This activity will highlight the complexities involved in such transactions."
6. Assignment – In-Depth Case Study:

Research and Analysis: "Select a UAE-based business group. Analyze a real or hypothetical asset transfer within this group and discuss the potential tax implications and necessary adjustments."
Conclusion:

Key Takeaways: "Today’s lecture underlined the importance of understanding intra-group transfers for accurate tax reporting. It’s crucial for businesses operating in corporate groups to handle such transactions with diligence."
Next Lecture Preview: "Our next session will focus on gains and losses not recognized in the income statement and their implications for corporate tax."

"""

if "model_messages" not in st.session_state:
    st.session_state.model_messages = []

# Initialize previous_input_type in session_state if it doesn't exist
if "previous_input_type" not in st.session_state:
    st.session_state.previous_input_type = None

# Check if the input type has changed
if st.session_state.previous_input_type != add_radio:
    # Clear the messages
    st.session_state.messages = []
    # Update previous_input_type
    st.session_state.previous_input_type = add_radio

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"][0])


if add_radio == 'Text ✏':
    model = genai.GenerativeModel(model_name="gemini-pro",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
    prompt = st.chat_input("Ask anything")

    if prompt:
        # Add user's prompt to the visible messages
        st.session_state.messages.append({
            "role": "user",
            "parts": [prompt]
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Combine the fixed instructions with the prompt for the model's input
        combined_input = fixed_instructions + prompt
        st.session_state.model_messages.append({
            "role": "user",
            "parts": [combined_input]
        })

        # Generate response from the model based on the combined input
        response = model.generate_content(st.session_state.model_messages)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(response.text)

        # Add the model's response to both message histories
        st.session_state.messages.append({
            "role": "model",
            "parts": [response.text]
        })
        st.session_state.model_messages.append({
            "role": "model",
            "parts": [response.text]
        })

elif add_radio == 'Image 📷':
    st.warning("Please upload an image and ask a question! Do not just send a text prompt, our model doesn't support that yet.", icon="🤖")
    model = genai.GenerativeModel('gemini-pro-vision',
                                generation_config=generation_config,
                                safety_settings=safety_settings)

    image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    prompt = st.chat_input("Ask anything")


    if image and prompt:
        st.session_state.messages = []
        # save image to buffer
        buffer = io.BytesIO()
        PIL.Image.open(image).save(buffer, format="JPEG") 
        image_input = PIL.Image.open(buffer)
        st.session_state.messages.append({
            "role":"user",
            "parts":[image_input],
        })
        with st.chat_message("user"):
            st.image(image_input, width=300)
            st.markdown(prompt)
        response = model.generate_content(st.session_state.messages)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown(response.text)
        st.session_state.messages.append({
            "role":"model",
            "parts":[response.text],
        })
