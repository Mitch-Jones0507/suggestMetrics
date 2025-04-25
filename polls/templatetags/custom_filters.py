from django import template

register = template.Library()


@register.filter
def underscore_to_title(value):
    return str(value).replace('_', ' ').title()


@register.filter
def is_or_not(value):
    return "is" if value else "is not"
