# User study: negotiation messages of the assistant

This folder contains the material of the user study of Section 7.2 of the paper. Participants assess **seven messages** of the negotiation step of our approach in **six scenarios**, which cover every kind of negotiation that our approach handles: relaxing a budget, an availability, a single region, or a single provider (with two or three options), choosing among trade-off compositions, and a negotiation of two rounds.

The study takes about 15 to 20 minutes. Participants give their informed consent before starting.

## Procedure

For each message, participants

1. answer a short comprehension question about the message,
2. answer the assistant in their own words, as they would in a real conversation (one or two sentences),
3. indicate which option their answer refers to,
4. indicate whether they would accept one of the proposed options and, if not, which option they would prefer (optional),
5. rate the message on a five-point Likert scale (Strongly disagree to Strongly agree) for
   - **Clarity** (the message is easy to understand),
   - **Relevance of the options** (the proposed options are reasonable for my situation),
   - **Ease of decision** (it was easy to make my decision).

Each free-form answer is then given to the negotiation LLM of our approach, and the **interpretation accuracy** is the proportion of answers that it interprets as the option chosen by the participant (step 3).

## Background questions

English proficiency level (A1 to C2), education level or job, field of study, work, or research, familiarity with cloud computing (Not familiar to I use it in my work), and frequency of use of LLM-based assistants (Never to Daily).

## Scenarios and messages

### Message 1 (Scenario 1, budget or availability)

*Situation.* You need a virtual machine and a database in Europe. Together, they must cost at most 259.26 USD per month and reach an availability of at least 99.6252%.

*Message.* Your requirements cannot be met simultaneously. Here are the only two possible ways to adjust your setup to meet your needs:
A: Increase your monthly budget for the virtual machine and database from 259.26 USD to 314.42 USD. This would keep your system running with 99.85% availability.
B: Lower the required availability for the virtual machine and database from 99.6252% to 99.4005%. This would reduce your monthly cost to 204.10 USD while maintaining 99.40% availability.
Which option do you prefer?

*Comprehension question.* Which option keeps your monthly cost within 259.26 USD? (A / B / Neither)

### Message 2 (Scenario 2, budget or single region)

*Situation.* You need a virtual machine and a database in the same region. Together, they must cost at most 150 USD per month.

*Message.* Your requirements cannot be met simultaneously. Here are the only two possible ways to adjust your setup to meet your needs:
Option A: Increase your monthly budget from 150.00 USD to 160.00 USD.
Option B: Allow the virtual machine and database to be in different regions.
Which option do you prefer?

*Comprehension question.* Which option keeps both services in the same region? (A / B / Both)

### Message 3 (Scenario 3, availability or single provider)

*Situation.* You need a virtual machine and a database from the same provider. Together, they must reach an availability of at least 99.95%.

*Message.* Your requirements cannot be met simultaneously. Here are the only two possible ways to adjust your setup to meet your needs:
Option A: Lower the required availability for the virtual machine and the database from 99.95% to 99.94%.
Option B: No longer require the virtual machine and the database to be from the same provider.
Which option do you prefer?

*Comprehension question.* Which option still requires a single provider? (A / B / Both)

### Message 4 (Scenario 4, three relaxation options)

*Situation.* You need a virtual machine and a database in the same region. Together, they must cost at most 150 USD per month and reach an availability of at least 99.9%.

*Message.* Your requirements cannot be met simultaneously. Here are the only three possible ways to adjust your setup to meet your needs:
A: Increase your monthly budget from 150.00 USD to 165.00 USD.
B: Lower the required availability from 99.90% to 99.49%.
C: Allow the virtual machine and database to be in different regions.
Which option do you prefer?

*Comprehension question.* Which option keeps both your budget and your availability requirement? (A / B / C)

### Message 5 (Scenario 5, trade-off compositions)

*Situation.* You need a virtual machine and a database in London. Together, they must cost at most 220 USD per month and reach an availability of at least 99.85%.

*Message.* Your requirements can be met simultaneously with two possible setups for your services:
Option A: Cost 187.00 USD per month, availability 99.85%.
Option B: Cost 191.00 USD per month, availability 99.94%.
Which option do you prefer, or does one factor (cost or availability) matter more to you?

*Comprehension question.* Which option is more available? (A / B)

### Message 6 (Scenario 6, turn 1, three relaxation options)

*Situation.* You need a virtual machine and a database from the same provider. Together, they must cost at most 195 USD per month and reach an availability of at least 99.85%.

*Message.* Your requirements cannot be met simultaneously. Here are the only three possible ways to adjust your setup to meet your needs:
A: Increase your monthly budget from 195.00 USD to 200.00 USD.
B: Lower the required availability from 99.85% to 99.83%.
C: Allow the virtual machine and database to use different providers.
Which option do you prefer?

*Comprehension question.* Which option(s) still require a single provider? (A / B / A and B / None of them)

### Message 7 (Scenario 6, turn 2, trade-off compositions after a relaxation)

*Situation.* Suppose that you chose option C, namely to use different providers. With different providers, two setups now meet all your requirements, each better on a different aspect.

*Message.* Your requirements can be met with two possible setups:
Option A: Cost 180.00 USD per month, availability 99.89%.
Option B: Cost 190.00 USD per month, availability 99.92%.
Which option do you prefer, or does one factor (cost or availability) matter more to you?

*Comprehension question.* Which option is cheaper? (A / B)

## Final questions

- Expressing my choice to the assistant in my own words was easy (five-point Likert scale).
- In a real situation, I would rely on such an assistant when my cloud requirements cannot all be met, or when several setups meet them (five-point Likert scale).
- Further comments or suggestions on the messages of the assistant (optional).

## Responses

`responses_template.csv` gives the columns of the anonymized responses. Personal data (e-mail address and organization) are not published.
