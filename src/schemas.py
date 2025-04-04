"""
Schema definitions for the reconciliation application.
"""
from typing import Dict, List, Optional
from datetime import datetime

# Common data types
DATE_TYPE = "date"
STRING_TYPE = "string"
INTEGER_TYPE = "integer"
FLOAT_TYPE = "float"
BOOLEAN_TYPE = "boolean"

# Column rename mappings
COLUMN_RENAMES = {
    "orders": {
        "seller id": "seller_id",
        "warehouse id": "warehouse_id",
        "store order id": "store_order_id",
        "order release id": "order_release_id",
        "order line id": "order_line_id",
        "seller order id": "seller_order_id",
        "order id fk": "order_id_fk",
        "core_item_id": "core_item_id",
        "created on": "created_on",
        "style id": "style_id",
        "seller sku code": "seller_sku_code",
        "sku id": "sku_id",
        "myntra sku code": "myntra_sku_code",
        "size": "size",
        "vendor article number": "vendor_article_number",
        "brand": "brand",
        "style name": "style_name",
        "article type": "article_type",
        "article type id": "article_type_id",
        "order status": "order_status",
        "packet id": "packet_id",
        "seller packe id": "seller_packe_id",
        "courier code": "courier_code",
        "order tracking number": "order_tracking_number",
        "seller warehouse id": "seller_warehouse_id",
        "cancellation reason id fk": "cancellation_reason_id_fk",
        "cancellation reason": "cancellation_reason",
        "packed on": "packed_on",
        "fmpu date": "fmpu_date",
        "inscanned on": "inscanned_on",
        "shipped on": "shipped_on",
        "delivered on": "delivered_on",
        "cancelled on": "cancelled_on",
        "rto creation date": "rto_creation_date",
        "lost date": "lost_date",
        "return creation date": "return_creation_date",
        "final amount": "final_amount",
        "total mrp": "total_mrp",
        "discount": "discount",
        "coupon discount": "coupon_discount",
        "shipping charge": "shipping_charge",
        "gift charge": "gift_charge",
        "tax recovery": "tax_recovery",
        "city": "city",
        "state": "state",
        "zipcode": "zipcode",
        "is_ship_rel": "is_ship_rel"
    },
    "returns": {
        "order_release_id": "order_release_id",
        "order_line_id": "order_line_id",
        "return_type": "return_type",
        "return_date": "return_date",
        "packing_date": "packing_date",
        "delivery_date": "delivery_date",
        "ecommerce_portal_name": "ecommerce_portal_name",
        "sku_code": "sku_code",
        "invoice_number": "invoice_number",
        "packet_id": "packet_id",
        "hsn_code": "hsn_code",
        "product_tax_category": "product_tax_category",
        "currency": "currency",
        "customer_paid_amount": "customer_paid_amount",
        "postpaid_amount": "postpaid_amount",
        "prepaid_amount": "prepaid_amount",
        "mrp": "mrp",
        "total_discount_amount": "total_discount_amount",
        "shipping_case_s": "shipping_case",
        "total_tax_rate": "total_tax_rate",
        "igst_amount": "igst_amount",
        "cgst_amount": "cgst_amount",
        "sgst_amount": "sgst_amount",
        "tcs_amount": "tcs_amount",
        "tds_amount": "tds_amount",
        "commission_percentage": "commission_percentage",
        "minimum_commission": "minimum_commission",
        "platform_fees": "platform_fees",
        "total_commission": "total_commission",
        "total_commission_plus_tcs_tds_deduction": "total_commission_plus_tcs_tds_deduction",
        "total_logistics_deduction": "total_logistics_deduction",
        "shipping_fee": "shipping_fee",
        "fixed_fee": "fixed_fee",
        "pick_and_pack_fee": "pick_and_pack_fee",
        "payment_gateway_fee": "payment_gateway_fee",
        "total_tax_on_logistics": "total_tax_on_logistics",
        "article_level": "article_level",
        "shipment_zone_classification": "shipment_zone_classification",
        "customer_paid_amt": "customer_paid_amt",
        "total_settlement": "total_settlement",
        "total_actual_settlement": "total_actual_settlement",
        "amount_pending_settlement": "amount_pending_settlement",
        "prepaid_commission_deduction": "prepaid_commission_deduction",
        "prepaid_logistics_deduction": "prepaid_logistics_deduction",
        "prepaid_payment": "prepaid_payment",
        "postpaid_commission_deduction": "postpaid_commission_deduction",
        "postpaid_logistics_deduction": "postpaid_logistics_deduction",
        "postpaid_payment": "postpaid_payment",
        "return_id": "return_id"
    },
    "settlement": {
        "order_release_id": "order_release_id",
        "order_line_id": "order_line_id",
        "return_type": "return_type",
        "return_date": "return_date",
        "packing_date": "packing_date",
        "delivery_date": "delivery_date",
        "ecommerce_portal_name": "ecommerce_portal_name",
        "sku_code": "sku_code",
        "invoice_number": "invoice_number",
        "packet_id": "packet_id",
        "hsn_code": "hsn_code",
        "product_tax_category": "product_tax_category",
        "currency": "currency",
        "customer_paid_amount": "customer_paid_amount",
        "postpaid_amount": "postpaid_amount",
        "prepaid_amount": "prepaid_amount",
        "mrp": "mrp",
        "total_discount_amount": "total_discount_amount",
        "shipping_case": "shipping_case",
        "total_tax_rate": "total_tax_rate",
        "igst_amount": "igst_amount",
        "cgst_amount": "cgst_amount",
        "sgst_amount": "sgst_amount",
        "tcs_amount": "tcs_amount",
        "tds_amount": "tds_amount",
        "commission_percentage": "commission_percentage",
        "minimum_commission": "minimum_commission",
        "platform_fees": "platform_fees",
        "total_commission": "total_commission",
        "total_commission_plus_tcs_tds_deduction": "total_commission_plus_tcs_tds_deduction",
        "total_logistics_deduction": "total_logistics_deduction",
        "shipping_fee": "shipping_fee",
        "fixed_fee": "fixed_fee",
        "pick_and_pack_fee": "pick_and_pack_fee",
        "payment_gateway_fee": "payment_gateway_fee",
        "total_tax_on_logistics": "total_tax_on_logistics",
        "article_level": "article_level",
        "shipment_zone_classification": "shipment_zone_classification",
        "customer_paid_amt": "customer_paid_amt",
        "total_settlement": "total_settlement",
        "total_actual_settlement": "total_actual_settlement",
        "amount_pending_settlement": "amount_pending_settlement",
        "prepaid_commission_deduction": "prepaid_commission_deduction",
        "prepaid_logistics_deduction": "prepaid_logistics_deduction",
        "prepaid_payment": "prepaid_payment",
        "postpaid_commission_deduction": "postpaid_commission_deduction",
        "postpaid_logistics_deduction": "postpaid_logistics_deduction",
        "postpaid_payment": "postpaid_payment",
        "return_id": "return_id"
    }
}

# Orders Schema
ORDERS_SCHEMA = {
    "seller_id": {"type": INTEGER_TYPE, "required": True},
    "warehouse_id": {"type": INTEGER_TYPE, "required": True},
    "store_order_id": {"type": STRING_TYPE, "required": True},
    "order_release_id": {"type": STRING_TYPE, "required": True},
    "order_line_id": {"type": STRING_TYPE, "required": True},
    "seller_order_id": {"type": STRING_TYPE, "required": True},
    "order_id_fk": {"type": STRING_TYPE, "required": True},
    "core_item_id": {"type": STRING_TYPE, "required": True},
    "created_on": {"type": DATE_TYPE, "required": True},
    "style_id": {"type": INTEGER_TYPE, "required": True},
    "seller_sku_code": {"type": STRING_TYPE, "required": True},
    "sku_id": {"type": INTEGER_TYPE, "required": True},
    "myntra_sku_code": {"type": STRING_TYPE, "required": True},
    "size": {"type": STRING_TYPE, "required": True},
    "vendor_article_number": {"type": STRING_TYPE, "required": True},
    "brand": {"type": STRING_TYPE, "required": True},
    "style_name": {"type": STRING_TYPE, "required": True},
    "article_type": {"type": STRING_TYPE, "required": True},
    "article_type_id": {"type": INTEGER_TYPE, "required": True},
    "order_status": {"type": STRING_TYPE, "required": True},
    "packet_id": {"type": STRING_TYPE, "required": True},
    "seller_packe_id": {"type": INTEGER_TYPE, "required": True},
    "courier_code": {"type": STRING_TYPE, "required": True},
    "order_tracking_number": {"type": STRING_TYPE, "required": True},
    "seller_warehouse_id": {"type": INTEGER_TYPE, "required": True},
    "cancellation_reason_id_fk": {"type": STRING_TYPE, "required": False},
    "cancellation_reason": {"type": STRING_TYPE, "required": False},
    "packed_on": {"type": DATE_TYPE, "required": False},
    "fmpu_date": {"type": DATE_TYPE, "required": False},
    "inscanned_on": {"type": DATE_TYPE, "required": False},
    "shipped_on": {"type": DATE_TYPE, "required": False},
    "delivered_on": {"type": DATE_TYPE, "required": False},
    "cancelled_on": {"type": DATE_TYPE, "required": False},
    "rto_creation_date": {"type": DATE_TYPE, "required": False},
    "lost_date": {"type": DATE_TYPE, "required": False},
    "return_creation_date": {"type": DATE_TYPE, "required": False},
    "final_amount": {"type": FLOAT_TYPE, "required": True},
    "total_mrp": {"type": FLOAT_TYPE, "required": True},
    "discount": {"type": FLOAT_TYPE, "required": True},
    "coupon_discount": {"type": FLOAT_TYPE, "required": True},
    "shipping_charge": {"type": FLOAT_TYPE, "required": True},
    "gift_charge": {"type": FLOAT_TYPE, "required": True},
    "tax_recovery": {"type": FLOAT_TYPE, "required": True},
    "city": {"type": STRING_TYPE, "required": True},
    "state": {"type": STRING_TYPE, "required": True},
    "zipcode": {"type": STRING_TYPE, "required": True},
    "is_ship_rel": {"type": INTEGER_TYPE, "required": True},
    "upload_timestamp": {"type": DATE_TYPE, "required": True}
}

# Returns Schema
RETURNS_SCHEMA = {
    "order_release_id": {"type": STRING_TYPE, "required": True},
    "order_line_id": {"type": STRING_TYPE, "required": True},
    "return_type": {"type": STRING_TYPE, "required": True},
    "return_date": {"type": DATE_TYPE, "required": True},
    "packet_id": {"type": STRING_TYPE, "required": True},
    "customer_paid_amount": {"type": FLOAT_TYPE, "required": True},
    "total_settlement": {"type": FLOAT_TYPE, "required": True},
    "total_actual_settlement": {"type": FLOAT_TYPE, "required": True},
    "amount_pending_settlement": {"type": FLOAT_TYPE, "required": True},
    "prepaid_commission_deduction": {"type": FLOAT_TYPE, "required": True},
    "prepaid_logistics_deduction": {"type": FLOAT_TYPE, "required": True},
    "prepaid_payment": {"type": FLOAT_TYPE, "required": True},
    "postpaid_commission_deduction": {"type": FLOAT_TYPE, "required": True},
    "postpaid_logistics_deduction": {"type": FLOAT_TYPE, "required": True},
    "postpaid_payment": {"type": FLOAT_TYPE, "required": True},
    "return_id": {"type": STRING_TYPE, "required": True},
    "upload_timestamp": {"type": DATE_TYPE, "required": True}
}

# Settlement Schema
SETTLEMENT_SCHEMA = {
    "order_release_id": {"type": STRING_TYPE, "required": True},
    "order_line_id": {"type": STRING_TYPE, "required": True},
    "return_type": {"type": STRING_TYPE, "required": True},
    "return_date": {"type": DATE_TYPE, "required": True},
    "packet_id": {"type": STRING_TYPE, "required": True},
    "customer_paid_amount": {"type": FLOAT_TYPE, "required": True},
    "total_settlement": {"type": FLOAT_TYPE, "required": False},
    "total_actual_settlement": {"type": FLOAT_TYPE, "required": False},
    "amount_pending_settlement": {"type": FLOAT_TYPE, "required": False},
    "prepaid_commission_deduction": {"type": FLOAT_TYPE, "required": False},
    "prepaid_logistics_deduction": {"type": FLOAT_TYPE, "required": False},
    "prepaid_payment": {"type": FLOAT_TYPE, "required": False},
    "postpaid_commission_deduction": {"type": FLOAT_TYPE, "required": False},
    "postpaid_logistics_deduction": {"type": FLOAT_TYPE, "required": False},
    "postpaid_payment": {"type": FLOAT_TYPE, "required": False},
    "return_id": {"type": STRING_TYPE, "required": False},
    "upload_timestamp": {"type": DATE_TYPE, "required": True}
} 