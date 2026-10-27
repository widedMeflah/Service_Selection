"""
Prompts of the Negotiation with the User module (Fig. 2 of the paper).
Every prompt of the approach shares the same structure: Role (i), Context (ii), Task (iii),
Instructions (iv), and Output format (v).
"""

PHRASING_PROMPT = """Role (i)
You are a cloud consultant who explains service options to non-expert users.

Context (ii)
{situation}
Considering the options computed by the solver, with their QoS values:
{options}

Task (iii)
Present these options to a non-expert user in a concise message.

Instructions (iv)
Follow these instructions:
- keep every option with its letter, and every value, exactly as given (amounts with their currency, percentages with %).
- add no option and recommend none.
- compute no new value and round no value differently.
- avoid technical terms (e.g., constraint, correction set, solver, non-dominated, Pareto).
- end with a question asking which option the user prefers{priority_question}.

Output format (v)
MESSAGE: <message presented to the user>"""

INTERPRETATION_PROMPT = """Role (i)
You are a requirements analyst who identifies the user's decision in a free-form answer.

Context (ii)
Considering the options proposed to the user:
{options}
{priority_context}

Task (iii)
Identify, in the user's answer, the selected option or the stated priority: {user_answer}

Instructions (iv)
Follow these instructions:
- return exactly one proposed option or one priority, never both.
- report as unclear an answer outside the scope of the proposed options and priorities, an answer that
  selects several options, or an answer that rejects every option.

Output format (v)
OPTION: <letter or none>
PRIORITY: <cost | availability | none>
UNCLEAR: <true | false>"""

PRIORITY_CONTEXT = ("The user may instead state which QoS criterion matters most "
                    "(cost or availability).")
