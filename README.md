# Trace_similarity_LLM
This project seeks to assess the capabilities of Large Language Models (LLMs) in the task of Entity Resolution.
The primary goal of the experiment is to verify whether the LLM can discern if an execution trace in XES format of a business process matches another one.

## Installing Requirements

To install the required Python packages for this project, you can use *pip* along with the *requirements.txt* file.

First, you need to clone the repository:
```bash
git clone https://github.com/AngeloC99/Trace_similarity_LLM
cd Trace_similarity_LLM
```

Run the following command to install the necessary dependencies using pip:
```bash
pip install -r requirements.txt
```

This command will read the requirements.txt file and install all the specified packages along with their dependencies.

## GPU Requirements
Please note that this software leverages open-source LLMs such as [Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) and [Llama 2](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf), which have specific requirements in terms of GPU availability.
It is recommended to have access to a GPU-enabled environment to run the software effectively.

# To Do
- [ ] Do tests, cases to consider for Llama and Mistral:
  - One shot;
  - Few shots;
  - Task description.
- [ ] Create slides with motivations of the work at the beginning

# Extensions
- Do the work done for a single trace to all the possible traces.

# Exam Guidelines
**Deadline**: April 30st (soft)
**Submission format**: Email with _documents_, _slides_ and/or _Github link_ with reproducibility instructions.

**Guidelines**: Aim for individual effort in your project, yet feel free to discuss and share ideas and resources with colleagues.
Be creative and incorporate concepts and datasets that are most interesting for you and/or your research.
Demonstrate critical knowledge of course topics and offer original insights.

After submitting the project we will schedule presentations for all the students.