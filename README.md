This repository is part of a side study done for my bachelor thesis research about the Degree of Specialization in LLM agent systems. 

The code in this repository is testing using an LLM-as-a-judge to find the percentage of domain knowledge and to catgorise references in the output. As can be read in my thesis, the experiment showed that this method is yet too unreliable. 

Another thing that can be found in this repository is the automatic calculation of the percentage of domain knowledge. The files have "\hl{" to mark the beginning of integrated domain knowledge and end with "}".

DoS0 are the results of the experiment with a degree of specialization that is zero.
DoS2 has three runs and one additional experiment (HITL = human-in-the-loop)
DOS4 has one run. 

The architecture is setup with CrewAI. 
