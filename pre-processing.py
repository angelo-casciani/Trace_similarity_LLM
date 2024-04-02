import xml.etree.ElementTree as ET


def replace_text_in_file(input_file, output_file, change):
    with open(input_file, 'r') as f:
        content = f.read()

    # Perform replacement for each key-value pair in the change dictionary
    for old_text, new_text in change.items():
        content = content.replace(old_text, new_text)

    with open(output_file, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    input_file_path = "data/food_delivery.xes"
    output_file_path = "data/food_delivery.xes"

    """
    change = {
        "customer order": "Customer Order Received",
        "recover client info": "Recover Customer Information",
        "check payment": "Check Payment",
        "create an account": "Create an Account",
        "check credit card and the payment": "Check Credit Card and Payment",
        "register the order": "Register the Order",
        "send confirmation to rider": "Send Confirmation to Rider",
        "notification order retired": "Receive Notification Order Retired",
        "select rider and create work proposal": "Select Rider and Create Work Proposal",
        "notification order retired to the client": "Send Notification Order Retired to Customer",
        "notification order complete": "Receive Complete Order Confirmation",
        "the rider refused the order": "Receive Refusal from Rider",
        "the rider accepted the order": "Receive Confirmation from Rider",
        "send to customer the waiting time": "Send Waiting Time to Customer",
        "send work proposal": "Send Work Proposal to Rider",
        "Estimated Arrival Time": "Estimate Waiting Time",
        "receive Customer Satisfaction": "Receive Customer Satisfaction Questionnaire",
        "pay rider": "Pay Rider",
        "order completed": "Order Completed"
    }"""

    change = {
        "Customer Order Received": "",
        "Recover Customer Information": "",
        "Check Payment": "",
        "Create an Account": "",
        "Check Credit Card and Payment": "",
        "Register the Order": "",
        "Send Confirmation to Rider": "",
        "Receive Notification Order Retired": "",
        "Select Rider and Create Work Proposal": "",
        "Send Notification Order Retired to Customer": "",
        "Receive Complete Order Confirmation": "",
        "Receive Refusal from Rider": "",
        "Receive Confirmation from Rider": "",
        "Send Waiting Time to Customer": "",
        "Send Work Proposal to Rider": "",
        "Estimate Waiting Time": "",
        "Receive Customer Satisfaction Questionnaire": "",
        "Pay Rider": "",
        "Order Completed": ""
    }

    replace_text_in_file(input_file_path, output_file_path, change)

    print("Replacement completed. Output file saved at:", output_file_path)
