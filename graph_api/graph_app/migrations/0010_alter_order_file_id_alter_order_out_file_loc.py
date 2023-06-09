# Generated by Django 4.1.5 on 2023-03-06 13:49

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('graph_app', '0009_order_out_file_loc_alter_order_file_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='order',
            name='file_id',
            field=models.UUIDField(blank=True, default=uuid.UUID('97d4f42f-2a19-4de9-8d91-29382b93c211'), editable=False, primary_key=True, serialize=False),
        ),
        migrations.AlterField(
            model_name='order',
            name='out_file_loc',
            field=models.CharField(blank=True, default=None, max_length=300, null=True),
        ),
    ]
