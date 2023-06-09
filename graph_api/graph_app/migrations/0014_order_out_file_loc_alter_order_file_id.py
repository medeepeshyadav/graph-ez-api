# Generated by Django 4.1.5 on 2023-03-06 14:00

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('graph_app', '0013_alter_order_file_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='order',
            name='out_file_loc',
            field=models.CharField(default=None, max_length=300),
        ),
        migrations.AlterField(
            model_name='order',
            name='file_id',
            field=models.UUIDField(blank=True, default=uuid.UUID('a0f2fe0f-c978-4b06-885c-8f89dc932eae'), editable=False, primary_key=True, serialize=False),
        ),
    ]
