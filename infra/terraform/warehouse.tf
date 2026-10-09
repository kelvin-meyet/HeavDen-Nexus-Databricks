# One serverless SQL warehouse, shared by all three environments (like the catalogs). Catalog
# Explorer's sample data, the SQL editor, AI/BI dashboards and Genie run on it; jobs and pipelines
# don't (they use serverless jobs compute). It bills only while running (2X-Small ~ $2.80/hour)
# and stops after 5 idle minutes.
resource "databricks_sql_endpoint" "shared" {
  name                      = "heavden-sql"
  cluster_size              = "2X-Small"
  min_num_clusters          = 1
  max_num_clusters          = 1 # no scale-out: one person queries it
  auto_stop_mins            = 5
  enable_serverless_compute = true
  warehouse_type            = "PRO" # serverless warehouses are PRO
  no_wait                   = true  # don't keep it running just to wait for it to start

  tags {
    custom_tags {
      key   = "project"
      value = "heavden"
    }
  }
}
