import json
import re

from typing import List
from fire import Fire
from pydantic import BaseModel

from inference import VLLMModel
from utils import find_math_answer


class Demonstration(BaseModel):
    texts: List[str]

    def make_prompt(self, prompt: str) -> str:
        raise NotImplementedError

    def get_stopping_words(self) -> List[str]:
        raise NotImplementedError

    def extract_answer(self, text: str) -> str:
        raise NotImplementedError


class GSM8KDemonstration(Demonstration):
    texts: List[str] = [
        "Question: There are 180 days in a school year.  A senior can skip their final exams if they miss 5% or less of the school year.  Hazel has missed 6 days of school due to illness.  How many more days can she miss and still not have to take her exams?",
        "Answer: There are 180 days in the school year and she can miss up to 5% so that's 180*.05 = 9 days\nHazel has been sick 6 days already and she can only miss 9 days or less so she can miss 9-6 = 3 more days. So the answer is \\boxed{3} days.",
        "Question: Several birds were sitting in the branches of a crape myrtle tree.  There were three times more cardinals than bluebirds, but half as many swallows as bluebirds. If there were 2 swallows, what is the total number of birds in the crape myrtle tree?",
        "Answer: With half as many swallows as bluebirds, there are 2*2=4 bluebirds.\nWith three times more cardinals than bluebirds, there are 3*4=12 cardinals,\nIf there were 2 swallows, then the total number of birds in the crape myrtle tree is 2+4+12=18 birds. So the answer is \\boxed{18}.",
        "Question: Barry goes to a shop to buy a shirt he'd been admiring for quite some time. He tells the attendant that it's his birthday so she decides to give him a 15% special discount. The price tag on the shirt says $80. How much is he supposed to pay now, considering the special discount?",
        "Answer: 15% of $80 = (15/100)*$80 = $12\nThe dollar amount of the discount is $12 so he is supposed to pay just $80-$12 = $68. So the answer is \\boxed{$68}.",
        "Question: Nancy wanted to make peanut butter cookies for a family gathering, but her cousin is allergic to peanuts. She decided to make almond butter cookies instead. A jar of almond butter costs three times the amount that a jar of peanut butter does. It takes half a jar to make a batch of cookies. A jar of peanut butter costs $3. How many dollars more does it cost per batch to make almond butter cookies instead of peanut butter cookies?",
        "Answer: A jar of almond butter costs 3 * 3 = $9.\nIt takes half a jar to make a batch of cookies, so it costs 9 / 2 = $4.50 to use almond butter.\nIt costs 3 / 2 = $1.50 to use peanut butter.\nThus, it costs 4.50 - 1.50 = $3 more to make a batch of almond butter cookies than peanut butter cookies. So the answer is \\boxed{$3}.",
    ]

    def make_prompt(self, prompt: str) -> str:
        sep = "\n\n"
        return sep.join(self.texts) + sep + f"Question: {prompt}{sep}Answer:"

    def get_stopping_words(self) -> List[str]:
        return ["\n\nQuestion:"]

    def extract_answer(self, text: str) -> str:
        if "\\boxed{" in text:
            text = text.split("\\boxed{")[-1].split("}")[0]

        filtered = "".join([char for char in text if char.isdigit() or char == " "])
        if not filtered.strip():
            return text
        return re.findall(pattern=r"\d+", string=filtered)[-1]


class MathDemonstration(Demonstration):
    # 4-Shot Demonstration from "Solving Quantitative Reasoning Problems with Language Models"
    # Also following llama-3 evaluation: https://github.com/meta-llama/llama3/blob/main/eval_details.md
    texts: List[str] = [
        "Question: Find the domain of the expression $\\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}",
        "Answer: The expressions inside each square root must be non-negative. Therefore, $x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of the expression is $\\boxed{[2,5)}$.\nSo the final answer is \\boxed{[2,5)}.",
        "Question: If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$",
        "Answer: We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \\boxed{24}.$\nSo the final answer is \\boxed{24}.",
        "Question: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
        "Answer: If Terrell lifts two 20-pound weights 12 times, he lifts a total of $2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$ pounds of weight. Equating this to 480 pounds, we can solve for $n$:\n\\begin{align*}\n30n&=480\\\\\n\Rightarrow\qquad n&=480/30=\\boxed{16}\n\end{align*}\nSo the final answer is \\boxed{16}.",
        "Question: If the system of equations\n\\begin{align*}\n6x-4y&=a,\\\\\n6y-9x &=b.\n\end{align*}\nhas a solution $(x, y)$ where $x$ and $y$ are both nonzero, find $\\frac{a}{b},$ assuming $b$ is nonzero.",
        "Answer: If we multiply the first equation by $-\\frac{3}{2}$, we obtain $$6y-9x=-\\frac{3}{2}a.$$\nSince we also know that $6y-9x=b$, we have $$-\\frac{3}{2}a=b\Rightarrow\\frac{a}{b}=\\boxed{-\\frac{2}{3}}.$$\nSo the final answer is \\boxed{-\\frac{2}{3}}.",
    ]

    def make_prompt(self, prompt: str) -> str:
        sep = "\n\n"
        return sep.join(self.texts) + sep + f"Question: {prompt}{sep}Answer:"

    def get_stopping_words(self) -> List[str]:
        return ["\n\nQuestion:"]

    def extract_answer(self, text: str) -> str:
        if "boxed" not in text:
            return text
        return find_math_answer(text)


class MMLUDemonstration(Demonstration):
    texts: List[str] = [
        "Question: All of the following statements about muscle contraction are true EXCEPT:\n(A) The ends of actin filaments move closer together.\n(B) The length of myosin filaments does not change.\n(C) Calcium-troponin binding precedes actin-myosin binding.\n(D) Calcium-tropomyosin binding precedes actin-myosin binding.",
        "Answer: To determine which statement about muscle contraction is not true, let's analyze each statement step by step: (A) The ends of actin filaments move closer together. - During muscle contraction, the actin filaments slide past the myosin filaments, causing the sarcomere (the functional unit of a muscle fiber) to shorten. This means the ends of the actin filaments do indeed move closer together. This statement is true. (B) The length of myosin filaments does not change. - The myosin filaments themselves do not change length during muscle contraction. Instead, they pull the actin filaments towards the center of the sarcomere. This statement is true. (C) Calcium-troponin binding precedes actin-myosin binding. - When a muscle cell is stimulated to contract, calcium ions are released from the sarcoplasmic reticulum. These calcium ions bind to troponin, causing a conformational change that moves tropomyosin away from the myosin-binding sites on actin filaments. This allows actin and myosin to bind and form cross-bridges. This statement is true. (D) Calcium-tropomyosin binding precedes actin-myosin binding. - Calcium ions do not bind directly to tropomyosin. Instead, they bind to troponin, which then causes tropomyosin to move away from the myosin-binding sites on actin. Therefore, calcium-tropomyosin binding does not occur. This statement is false. Based on this analysis, the statement that is not true is: (D) Calcium-tropomyosin binding precedes actin-myosin binding.",
        "Question: Irrelevant thoughts that provide stimulation when your interest is flagging, letting you experience positive emotions, are\n(A) nonconscious\n(B) unconscious\n(C) daydreams\n(D) delusions",
        "Answer: Let's break down the options one by one to determine the best answer. (A) Nonconscious: This term refers to processes that occur in the mind without conscious awareness. While nonconscious processes can influence thoughts and behaviors, they are not typically characterized by providing stimulation or positive emotions in a direct, experiential way. (B) Unconscious: Similar to nonconscious, unconscious processes are those that occur without our awareness. These can include repressed memories or instincts, but they are not typically described as providing direct stimulation or positive emotions when interest is flagging. (C) Daydreams: Daydreams are a form of spontaneous, self-generated thought that often occurs when a person is not fully engaged in their current activity. They can provide mental stimulation and positive emotions, as they allow the mind to wander to pleasant or interesting scenarios. This fits well with the description given in the question. (D) Delusions: Delusions are false beliefs that are strongly held despite evidence to the contrary, often associated with mental health conditions. They are not typically described as providing stimulation or positive emotions in the context of flagging interest. Given the descriptions, the best answer is: (C) Daydreams. Daydreams are indeed irrelevant thoughts that can provide stimulation and positive emotions when your interest is flagging.",
        'Question: Six-month-old Sasha loves to play "peek-a-boo" with her mother, an indication that she has developed a sense of\n(A) play versus learning\n(B) transitivity\n(C) metacognition\n(D) object permanence',
        "Answer: To determine which concept is indicated by Sasha's enjoyment of 'peek-a-boo,' let's analyze each option step by step: (A) Play versus learning: This concept involves distinguishing between activities that are purely for fun (play) and those that are intended for acquiring knowledge or skills (learning). At six months old, Sasha is likely not yet capable of making this distinction. (B) Transitivity: This is a logical concept that involves understanding relationships between different elements in a sequence. For example, if A is greater than B, and B is greater than C, then A is greater than C. This is a more advanced cognitive skill that typically develops later in childhood. (C) Metacognition: This refers to the awareness and understanding of one's own thought processes. It involves higher-order thinking skills, such as planning, monitoring, and evaluating one's own learning and problem-solving strategies. This is also a more advanced cognitive ability that develops later in childhood. (D) Object permanence: This is the understanding that objects continue to exist even when they cannot be seen, heard, or otherwise sensed. This concept typically begins to develop around 4 to 7 months of age. Playing 'peek-a-boo' is a classic example of a game that helps infants develop and demonstrate their understanding of object permanence. Given these explanations, the most appropriate answer is: (D) Object permanence.",
        "Question: Which of the following statements most closely corresponds with differential association theory?\n(A) If all of your friends jumped off a bridge, I suppose you would too.\n(B) You should be proud to be a part of this organization.\n(C) If the door is closed, try the window.\n(D) Once a thief, always a thief.",
        "Answer: To determine which statement most closely corresponds with differential association theory, let's first understand what differential association theory is. This theory, developed by Edwin H. Sutherland, posits that criminal behavior is learned through interactions with others. According to this theory, individuals learn values, attitudes, techniques, and motives for criminal behavior through their associations with others. Now, let's analyze each statement in the context of differential association theory: (A) 'If all of your friends jumped off a bridge, I suppose you would too.' - This statement suggests that an individual might engage in certain behaviors because their friends are doing so. It implies that behavior is influenced by the actions of one's social group, which aligns with the idea that behavior is learned through interactions with others. (B) 'You should be proud to be a part of this organization.' - This statement is about feeling pride in being part of a group or organization. It does not directly relate to learning behavior through social interactions, especially in the context of criminal behavior. (C) 'If the door is closed, try the window.' - This statement suggests finding alternative solutions or being resourceful. It does not relate to learning behavior through social interactions. (D) 'Once a thief, always a thief.' - This statement implies that once someone has engaged in criminal behavior, they will always continue to do so. It suggests a fixed character trait rather than behavior learned through social interactions. Given this analysis, the statement that most closely corresponds with differential association theory is: (A) 'If all of your friends jumped off a bridge, I suppose you would too.' This statement aligns with the idea that individuals learn and adopt behaviors based on their interactions and associations with others.",
    ]

    def make_prompt(self, prompt: str) -> str:
        sep = "\n\n"
        return sep.join(self.texts) + sep + f"Question: {prompt}{sep}Answer:"

    def get_stopping_words(self) -> List[str]:
        return ["Question:"]

    def extract_answer(self, text: str) -> str:
        letters = "ABCD"
        matches = re.findall(f"\(([{letters}])\)", text)
        if matches:
            return "(" + matches[-1] + ")"

        matches = re.findall(f"[{letters}]", text)
        if matches:
            return "(" + matches[-1] + ")"

        return text


class MMLUStemDemonstration(Demonstration):
    texts: List[str] = [
        "Question: Which of the following adaptive features would least likely be found in an animal living in a hot arid environment?\n(A) Long loops of Henle to maximize water reabsorption\n(B) Storage of water in fatty tissues\n(C) Large ears to aid in heat dispersion\n(D) Short loops of Henle to maximize water secretion",
        "Answer: To determine which adaptive feature would be least likely found in an animal living in a hot arid environment, let's analyze each option in the context of such an environment: (A) Long loops of Henle to maximize water reabsorption: In a hot arid environment, water is scarce. Animals would benefit from adaptations that help conserve water. Long loops of Henle in the kidneys are an adaptation that allows for maximum reabsorption of water, which is crucial for survival in such conditions. This feature is likely to be found in animals living in arid environments. (B) Storage of water in fatty tissues: Some animals have evolved to store water in their bodies to survive long periods without access to water. Storing water in fatty tissues is an effective way to ensure a supply of water during droughts or dry periods. This is another adaptation that would be beneficial in a hot arid environment. (C) Large ears to aid in heat dispersion: In hot environments, animals need to dissipate excess body heat to avoid overheating. Large ears can help with this by increasing the surface area through which heat can be lost. This adaptation is also likely to be found in animals living in hot arid environments. (D) Short loops of Henle to maximize water secretion: Short loops of Henle would result in less water reabsorption and more water being excreted. In a hot arid environment where water is scarce, this would be a disadvantage because it would lead to greater water loss. Therefore, this feature is least likely to be found in animals living in such conditions. So the correct answer option is (D) Short loops of Henle to maximize water secretion",
        "Question: A survey of 1000 registered voters revealed that 450 people would vote for candidate A in an upcoming election. If 220,000 people vote in the election, how many votes would the survey takers predict candidate A should receive?\n(A) 44,500\n(B) 48,900\n(C) 95,000\n(D) 99,000",
        "Answer: To predict the number of votes candidate A should receive based on the survey, we can use a proportion. The survey indicates that 450 out of 1000 registered voters would vote for candidate A. We can set up the following proportion to find the predicted number of votes out of 220,000 voters: \\[\\frac{450}{1000} = \\frac{x}{220,000}\\] To solve for \\( x \\), we can cross-multiply and then divide: \\[450 \\times 220,000 = 1000 \\times x\\] \\[99,000,000 = 1000x\\] \\[x = \\frac{99,000,000}{1000}\\] \\[x = 99,000\\] Therefore, the survey takers would predict that candidate A should receive \\( 99,000 \\) votes. So the correct answer option is (D) 99,000",
        "Question: A cannonball is fired horizontally at 10 m/s from a cliff. Its speed one second after being fired is about\n(A) 10 m/s\n(B) 14 m/s\n(C) 16 m/s\n(D) 20 m/s",
        "Answer: To determine the speed of the cannonball one second after being fired horizontally, we need to consider both the horizontal and vertical components of its motion. 1. Horizontal Motion: - The horizontal velocity (v_x) remains constant because there is no horizontal acceleration (assuming air resistance is negligible). - Given: v_x = 10 m/s 2. Vertical Motion: - The vertical velocity (v_y) changes due to the acceleration caused by gravity (g = 9.8 m/s^2). - Initially, the vertical velocity is zero because the cannonball is fired horizontally. - After one second, the vertical velocity can be calculated using the equation: v_y = g * t where t = 1 s. v_y = 9.8 m/s^2 * 1 s = 9.8 m/s 3. Resultant Speed: - The resultant speed (v) is found by combining the horizontal and vertical components using the Pythagorean theorem: v = sqrt(v_x^2 + v_y^2) v = sqrt((10 m/s)^2 + (9.8 m/s)^2) v = sqrt(100 + 96.04) v = sqrt(196.04) v ≈ 14 m/s Therefore, the speed of the cannonball one second after being fired is about 14 m/s. So the correct answer option is (B) 14 m/s",
        "Question: A 3-microfarad capacitor is connected in series with a 6-microfarad capacitor. When a 300-volt potential difference is applied across this combination, the total energy stored in the two capacitors is\n(A) 0.09 J\n(B) 0.18 J\n(C) 0.27 J\n(D) 0.41 J",
        "Answer: To find the total energy stored in the two capacitors connected in series, we need to follow these steps: 1. Calculate the equivalent capacitance of the series combination: For capacitors in series, the reciprocal of the equivalent capacitance \( C_{\text{eq}} \) is the sum of the reciprocals of the individual capacitances: \[ \frac{1}{C_{\text{eq}}} = \frac{1}{C_1} + \frac{1}{C_2} \] Given \( C_1 = 3 \, \mu\text{F} \) and \( C_2 = 6 \, \mu\text{F} \): \[ \frac{1}{C_{\text{eq}}} = \frac{1}{3 \, \mu\text{F}} + \frac{1}{6 \, \mu\text{F}} = \frac{2}{6 \, \mu\text{F}} + \frac{1}{6 \, \mu\text{F}} = \frac{3}{6 \, \mu\text{F}} = \frac{1}{2 \, \mu\text{F}} \] Therefore, \[ C_{\text{eq}} = 2 \, \mu\text{F} \] 2. Calculate the total energy stored in the equivalent capacitor: The energy \( E \) stored in a capacitor is given by: \[ E = \frac{1}{2} C V^2 \] Here, \( C = C_{\text{eq}} = 2 \, \mu\text{F} = 2 \times 10^{-6} \, \text{F} \) and \( V = 300 \, \text{V} \): \[ E = \frac{1}{2} \times 2 \times 10^{-6} \, \text{F} \times (300 \, \text{V})^2 \] \[ E = \frac{1}{2} \times 2 \times 10^{-6} \times 90000 \] \[ E = 1 \times 10^{-6} \times 90000 \] \[ E = 0.09 \, \text{J} \] Therefore, the total energy stored in the two capacitors is \( \boxed{0.09 \, \text{J}} \). So the correct answer option is (A) 0.09",
    ]

    def make_prompt(self, prompt: str) -> str:
        sep = "\n\n"
        return sep.join(self.texts) + sep + f"Question: {prompt}{sep}Answer:"

    def get_stopping_words(self) -> List[str]:
        return ["Question:", "<|eot_id|>"]

    def extract_answer(self, text: str) -> str:
        letters = "ABCD"
        matches = re.findall(f"\(([{letters}])\)", text)
        if matches:
            return "(" + matches[-1] + ")"

        matches = re.findall(f"[{letters}]", text)
        if matches:
            return "(" + matches[-1] + ")"

        return text


class CSQADemonstration(Demonstration):
    texts: List[str] = [
        "Question: If the though of performing makes you anxious what do you suffer from?\n(A) being appreciated\n(B) get attention\n(C) terror\n(D) stage fright\n(E) fear",
        "Answer: To determine what you suffer from if the thought of performing makes you anxious, let's analyze each option step by step: (A) being appreciated - This option does not fit because being appreciated is generally a positive experience and not typically associated with anxiety. (B) get attention - While getting attention can make some people anxious, it is not specific enough to describe the anxiety related to performing. (C) terror - Terror is a strong word for extreme fear, but it is not specific to performing. (D) stage fright - This term specifically refers to the anxiety or fear experienced when performing in front of an audience. It directly addresses the context of performing. (E) fear - Fear is a general term and does not specifically relate to performing. Given the context of performing and the specific nature of the anxiety described, the most accurate answer is: (D) stage fright. So the final answer is (D) stage fright.",
        "Question: Where would you put some pennies that you plan to use soon?\n(A) pocket\n(B) jar\n(C) water fountain\n(D) drawer\n(E) trunk",
        "Answer: To determine the best place to put some pennies that you plan to use soon, let's evaluate each option step by step: (A) Pocket: A pocket is a convenient and easily accessible place. You can quickly reach into your pocket to retrieve the pennies when needed. This option is practical for short-term use and immediate access. (B) Jar: A jar is typically used for longer-term storage or saving. It might not be as convenient for quick access compared to a pocket. You would need to open the jar and possibly sort through other coins to find the pennies. (C) Water Fountain: A water fountain is not a suitable place to store pennies. Pennies placed in a water fountain are usually considered a donation or wish, not for retrieval. This option is impractical and not intended for storing coins for future use. (D) Drawer: A drawer can be a good place for storage, but it may not be as immediately accessible as a pocket. You would need to go to the drawer and open it to get the pennies. This option is more suitable for medium-term storage rather than immediate use. (E) Trunk: A trunk is typically used for long-term storage or transporting items. It is not convenient for quick access to pennies. This option is impractical for pennies you plan to use soon. Based on the evaluation, the best place to put some pennies that you plan to use soon is: (A) pocket. This option provides the most convenience and immediate access for short-term use. So the final answer is (A) pocket.",
        "Question: They enjoyed the stories their uncle told, he was the eldest and the closest living connection to the what?\n(A) aunts house\n(B) aunt's house\n(C) books\n(D) most families\n(E) older generation",
        "Answer: To determine the correct answer, let's analyze the sentence step by step: 1. *Identify the context*: The sentence talks about stories told by an uncle who is the eldest and the closest living connection to something. 2. *Understand the relationship*: The uncle is described as the eldest, which implies a connection to something related to age or generational ties. 3. *Evaluate the options*: (A) aunts house: This option refers to a place, but it doesn't fit well with the idea of a generational connection. (B) aunt's house: Similar to (A), this option refers to a place and doesn't align with the generational context. (C) books: This option doesn't fit the context of being the eldest and a living connection. (D) most families: This option is too broad and doesn't specifically relate to the idea of being the eldest. (E) older generation: This option fits well because it directly relates to the idea of being the eldest and having a generational connection. 4. *Select the best fit*: Based on the analysis, the option that best fits the context of the sentence is (E) older generation. Therefore, the correct answer is: (E) older generation. So the final answer is (E) older generation.",
        "Question: In what state would you find the Shenandoah Valley?\n(A) west virginia\n(B) mountainous region\n(C) montana\n(D) continent\n(E) earth",
        "Answer: To determine the state where the Shenandoah Valley is located, let's analyze each option step by step: 1. *Option (A) West Virginia*: The Shenandoah Valley is not primarily located in West Virginia. While parts of the valley extend into West Virginia, the majority of it is in another state. 2. *Option (B) Mountainous region*: This is not a state. It describes a type of geographical area, but it does not specify a state. 3. *Option (C) Montana*: The Shenandoah Valley is not located in Montana. Montana is far to the west of the Shenandoah Valley. 4. *Option (D) Continent*: This is not a state. It refers to a large landmass, but it does not specify a state. 5. *Option (E) Earth*: This is not a state. It refers to the planet, but it does not specify a state. The Shenandoah Valley is actually located in the state of Virginia. However, since Virginia is not listed as an option, the closest correct answer based on the given choices would be: *Option (A) West Virginia*, as parts of the Shenandoah Valley do extend into West Virginia, even though the majority of it is in Virginia. So the final answer is (A) west virginia.",
    ]

    def make_prompt(self, prompt: str) -> str:
        sep = "\n\n"
        return sep.join(self.texts) + sep + f"Question: {prompt}{sep}Answer:"

    def get_stopping_words(self) -> List[str]:
        return ["Question:"]

    def extract_answer(self, text: str) -> str:
        letters = "ABCDE"
        matches = re.findall(f"\(([{letters}])\)", text)
        if matches:
            return "(" + matches[-1] + ")"

        matches = re.findall(f"[{letters}]", text)
        if matches:
            return "(" + matches[-1] + ")"

        return text


class WINOGRANDEDemonstration(Demonstration):
    texts: List[str] = [
        "Question: Painting the garage was a pleasure to Jeffrey but not to Justin, as _ loved the quiet nature of the chore.\n(1) Jeffrey\n(2) Justin",
        '''Answer: To determine the correct answer, we need to identify who enjoyed the quiet nature of the chore. Let's break down the sentence step by step: "Painting the garage was a pleasure to Jeffrey but not to Justin" - This part of the sentence tells us that Jeffrey enjoyed painting the garage, while Justin did not. "as _ loved the quiet nature of the chore." - This part of the sentence explains the reason why someone enjoyed painting the garage. The blank should be filled with the name of the person who loved the quiet nature of the chore. Since Jeffrey found painting the garage to be a pleasure, it implies that he enjoyed the quiet nature of the chore. Therefore, the correct answer is: (1) Jeffrey. So the complete sentence is: "Painting the garage was a pleasure to Jeffrey but not to Justin, as Jeffrey loved the quiet nature of the chore."''',
        "Question: Amy helped Emily immensely with her disability, as _ had so much help she offered.\n(1) Amy\n(2) Emily",
        '''Answer: To determine the correct pronoun to fill in the blank, we need to understand the context of the sentence and the roles of Amy and Emily. Amy is the one who helped. Emily is the one who received help. The pronoun should refer to the person who had a lot of help to offer. Amy is the one who helped, so she is the one who had a lot of help to offer. The correct pronoun should refer to Amy. Therefore, the correct answer is: (1) Amy. The completed sentence is: "Amy helped Emily immensely with her disability, as Amy had so much help she offered."''',
        "Question: Pat and Judy's two-family garage sale included baby toys and LPs. The _ were donated by Judy's aging parents.\n(1) toys\n(2) LPs",
        "Answer: To determine whether the baby toys or the LPs were donated by Judy's aging parents, let's analyze the context and the likely scenario: 1. *Baby Toys*: These are typically used by infants and young children. If Judy's aging parents donated these, it would imply that they had kept baby toys from when Judy or her siblings were young. While possible, it is less common for people to keep baby toys for such a long time. 2. *LPs (Long Play Records)*: These are vinyl records that were popular in the mid-20th century. Judy's aging parents would likely have been adults or young adults during the peak popularity of LPs. It is more plausible that they would have a collection of LPs that they no longer need or use. Given these considerations, it is more likely that the LPs were donated by Judy's aging parents. Therefore, the correct answer is: (2) LPs",
        "Question: Mona had a panic attack while public speaking and found out she was agoraphobic. The _ was stopped.\n(1) attack\n(2) speaking",
        "Answer: To determine the correct word to fill in the blank, let's analyze the sentence step by step: Agoraphobia is an anxiety disorder characterized by an intense fear of being in situations where escape might be difficult or help might not be available, often leading to avoidance of such situations. Given that Mona had a panic attack while public speaking, and considering the nature of agoraphobia, it is likely that the activity causing the anxiety (public speaking) was stopped. Now, let's evaluate the options: - *Option (1) attack*: This would imply that the panic attack itself was stopped. However, the sentence structure and context suggest that the focus is on the activity that triggered the panic attack. - *Option (2) speaking*: This implies that the public speaking was stopped, which aligns with the context of agoraphobia and the likely response to a panic attack. Therefore, the correct word to fill in the blank is: The *speaking* was stopped. So the final answer is (2) speaking.",
    ]

    def make_prompt(self, prompt: str) -> str:
        sep = "\n\n"
        return sep.join(self.texts) + sep + f"Question: {prompt}{sep}Answer:"

    def get_stopping_words(self) -> List[str]:
        return ["Question:"]

    def extract_answer(self, text: str) -> str:
        letters = "12"
        matches = re.findall(f"\(([{letters}])\)", text)
        if matches:
            return "(" + matches[-1] + ")"

        matches = re.findall(f"[{letters}]", text)
        if matches:
            return "(" + matches[-1] + ")"

        return text
    

class ZeroShotChatDemonstration(Demonstration):
    # This is the default template used in llama-factory for training
    texts: List[str] = []

    def make_prompt(self, prompt: str) -> str:
        return f"Human: {prompt}\nAssistant: "

    def get_stopping_words(self) -> List[str]:
        return ["Human:"]

    def extract_answer(self, text: str) -> str:
        filtered = "".join([char for char in text if char.isdigit() or char == " "])
        if not filtered.strip():
            return text
        return re.findall(pattern=r"\d+", string=filtered)[-1]


def select_demonstration(name: str, **kwargs):
    if name == "gsm8k":
        return GSM8KDemonstration()
    if name == "math":
        return MathDemonstration()
    if name == "mmlu":
        return MMLUDemonstration()
    if name == "mmlu_stem":
        return MMLUStemDemonstration()
    if name == "csqa":
        return CSQADemonstration()
    if name == "winogrande":
        return WINOGRANDEDemonstration()
    if name == "zero_chat":
        return ZeroShotChatDemonstration(**kwargs)
    raise KeyError(name)


def test_demo(
    name: str,
    question: str = "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
    **kwargs,
):
    demo = select_demonstration(name)
    model = VLLMModel(**kwargs)
    model.stopping_words = demo.get_stopping_words()
    prompt = demo.make_prompt(question)
    output = model.run(prompt)
    pred = demo.extract_answer(output)
    info = dict(question=question, prompt=prompt, output=output, pred=pred)
    print(json.dumps(info, indent=2))


def test_extract_answer(name: str = "math"):
    demo = select_demonstration(name)
    for text in demo.texts:
        if not text.startswith("Question"):
            print(dict(answer=text))
            print(dict(found_ans=demo.extract_answer(text)))
            print(dict(found_a_2=demo.extract_answer(demo.extract_answer(text))))


if __name__ == "__main__":
    Fire()
