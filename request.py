import requests

url = 'http://127.0.0.1:5000/predict' 

r = requests.post(url, json={'nama': 73, 'income_user': 0, 'address': 5, 'RT': 12, 'RW': 5, 'zip_code': 17, 'location_criminality': 2,
 'IP_address': 41, 'address_suitability': 1, 'amount_of_money_spent': 45, 'user_behavior': 0, 'last_user_behavior_recorded_date': 4,
 'worst_credit_performance': 1, 'last_12_months_credit_performance': 123000000000.0, 'last_12_months_cost_of_collection': 52,
 'suitability_of_the_company_credit_type_with_ company_prospects': 1, 'key_person_ID_existence': 1, 'company_reputation_crawl_results': 0,
 'caller_ID_name_on_the_core_of_company': 3, 'number_of_company_related_articles_on_google': 1, 'amount_of_principal_installment': 72,
 'character_of_the_owner_key_figures_in_media': 2, 'suitability_of_the_size_of_the _company_with_transaction_value': 1, 'existency_badan_usaha': 1,
 'all_document_validity': 4, 'falsification_of_signature_data_letter numbers': 1, 'type_of_business': 4, 'address_match': 1,
 'amount_of_principal_installment_dan_principal_interest': 77, 'company_address': 18, 'last_24_months_credit_perfomance': 0,
 'last_credit_survey_process': 6, 'business_prospects': 2, 'business_threat': 1, 'industry_business_name': 1, 'business_identification_number_industry': 26,
 'type_of_product': 0, 'average_sales_not_profit': 64, 'credit_scoring': 1, 'hardware_used': 18, 'loan_to_value': 1, 'tenor': 0,
 'besar_angsuran_pokok_dan_bunga': 27489007, 'last_contact_with_reply_or_mee_ in_days': 52, 'last_payment_time_days': 50})

print(r.json())