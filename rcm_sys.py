import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from pyspark.sql import SparkSession
from pyspark.ml.tuning import CrossValidatorModel
import pyspark.sql.functions as F
import os

# Set the path to Python 3.9 for Spark
os.environ["PYSPARK_PYTHON"] = "/Users/nguyenhoangvi/opt/anaconda3/bin/python3.9"
os.environ["PYSPARK_DRIVER_PYTHON"] = "/Users/nguyenhoangvi/opt/anaconda3/bin/python3.9"

# Initialize Spark session
spark = SparkSession.builder.appName("Sales Dashboard").getOrCreate()

st.set_page_config(page_title="Big Data Dashboard", page_icon=":bar_chart:", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Sales Overview", "Detailed Analysis"])

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv('OnlineRetail.csv', parse_dates=['InvoiceDate'], dayfirst=True, encoding='ISO-8859-1')
    data['TotalAmount'] = data['Quantity'] * data['UnitPrice']
    data.dropna(inplace=True)
    return data

data = load_data()

# Filter function
def apply_filters(data, page_id=""):
    invoice_filter = st.sidebar.multiselect("Invoice Number:", data['InvoiceNo'].unique(), key=f"invoice_{page_id}")
    customer_filter = st.sidebar.multiselect("Customer ID:", data['CustomerID'].unique(), key=f"customer_{page_id}")
    stock_filter = st.sidebar.multiselect("Stock Code:", data['StockCode'].unique(), key=f"stock_{page_id}")
    country_filter = st.sidebar.multiselect("Country:", data['Country'].unique(), key=f"country_{page_id}")

    if invoice_filter:
        data = data[data['InvoiceNo'].isin(invoice_filter)]
    if customer_filter:
        data = data[data['CustomerID'].isin(customer_filter)]
    if stock_filter:
        data = data[data['StockCode'].isin(stock_filter)]
    if country_filter:
        data = data[data['Country'].isin(country_filter)]

    return data

# Date filter
startDate, endDate = data["InvoiceDate"].min(), data["InvoiceDate"].max()
col1, col2 = st.columns(2)
with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))
with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

filtered_data = data[(data["InvoiceDate"] >= date1) & (data["InvoiceDate"] <= date2)].copy()
filtered_data = apply_filters(filtered_data)


if page == "Sales Overview":
    st.title(":bar_chart: Sample SuperStore EDA")
    st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)

        # Check if filtered_data is empty
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
    else:
        # Compute total sales by StockCode
        category_df = filtered_data.groupby(by=["StockCode"], as_index=False)["TotalAmount"].sum()

        # Display bar chart for sales by StockCode
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Doanh thu theo StockCode")
            display_stock_df = category_df.nlargest(10, "TotalAmount")
            fig = px.bar(display_stock_df, x="StockCode", y="TotalAmount",
                         text=['${:,.2f}'.format(x) for x in display_stock_df["TotalAmount"]],
                         template="seaborn")
            st.plotly_chart(fig, use_container_width=True)

        # Sales by Country
        country_data = filtered_data.groupby(by="Country", as_index=False)["TotalAmount"].sum()

        with col2:
            st.subheader("Doanh thu theo Quốc gia")
            display_country_df = country_data.nlargest(10, "TotalAmount")
            fig_country = px.pie(display_country_df, values="TotalAmount", names="Country", hole=0.5)
            fig_country.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_country, use_container_width=True)

        # Data viewing with download option
        cl1, cl2 = st.columns(2)

        with cl1:
            with st.expander("Category_ViewData"):
                st.write(category_df.style.background_gradient(cmap="Blues"))
                csv = category_df.to_csv(index=False).encode('utf-8')
                st.download_button("Tải dữ liệu", data=csv, file_name="Category.csv", mime="text/csv",
                                   help='Click here to download the data as a CSV file')

        with cl2:
            with st.expander("Country_ViewData"):
                st.write(country_data.style.background_gradient(cmap="Oranges"))
                csv_country = country_data.to_csv(index=False).encode('utf-8')
                st.download_button("Tải dữ liệu", data=csv_country, file_name="Country.csv", mime="text/csv",
                                   help='Click here to download the data as a CSV file')

        # Time Series Analysis
        filtered_data["month_year"] = filtered_data["InvoiceDate"].dt.to_period("M")
        st.subheader('Time Series Analysis')

        linechart = pd.DataFrame(filtered_data.groupby(filtered_data["month_year"].dt.strftime("%Y : %b"))["TotalAmount"].sum()).reset_index()
        linechart.columns = ["month_year", "Sales"]

        # Line chart
        fig2 = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"}, height=500, width=1000, template="gridon")
        st.plotly_chart(fig2, use_container_width=True)

        # Download data for Time Series
        with st.expander("View Data of TimeSeries:"):
            st.write(linechart.T.style.background_gradient(cmap="Blues"))
            csv = linechart.to_csv(index=False).encode("utf-8")
            st.download_button('Download Data', data=csv, file_name="TimeSeries.csv", mime='text/csv')


        # Segment charts
        chart1, chart2 = st.columns(2)

        # Limit the number of entries displayed in pie charts (e.g., top 10)
        top_customers = (
            filtered_data.groupby("CustomerID")["TotalAmount"]
            .sum()
            .nlargest(10)
            .reset_index()
        )

        top_stock_codes = (
            filtered_data.groupby("StockCode")["TotalAmount"]
            .sum()
            .nlargest(10)
            .reset_index()
        )

        with chart1:
            st.subheader('Top 10 CustomerID - TotalAmount')
            fig = px.pie(top_customers, values="TotalAmount", names="CustomerID", template="plotly_dark")
            fig.update_traces(textinfo='percent+label', textposition="inside")
            st.plotly_chart(fig, use_container_width=True)

        with chart2:
            st.subheader('Top 10 StockCode - TotalAmount')
            fig = px.pie(top_stock_codes, values="TotalAmount", names="StockCode", template="gridon")
            fig.update_traces(textinfo='percent+label', textposition="inside")
            st.plotly_chart(fig, use_container_width=True)

        # Summary Tables
        st.subheader(":point_right: Sales Summary")

        with st.expander("Summary Table"):
            df_sample = filtered_data[["Country", "StockCode", "Description", "TotalAmount", "UnitPrice", "Quantity"]].head(5)
            fig = ff.create_table(df_sample, colorscale="Cividis")
            st.plotly_chart(fig, use_container_width=True)

            # Download button for summary table
            csv = df_sample.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Summary Table as CSV",
                data=csv,
                file_name='summary_table.csv',
                mime='text/csv',
            )

            # Month-Sale Table
            st.markdown("Month-Sale-Table")
            filtered_data["month"] = filtered_data["InvoiceDate"].dt.month_name()
            sub_category_year = pd.pivot_table(
                data=filtered_data, 
                values="TotalAmount", 
                index=["Description"], 
                columns="month", 
                aggfunc="sum"
            )
            
            # Display with a gradient style
            st.write(sub_category_year.style.background_gradient(cmap="Blues"))

            # Download button for Month-Sale table
            month_sale_csv = sub_category_year.to_csv().encode('utf-8')
            st.download_button(
                label="Download Month-Sale Table as CSV",
                data=month_sale_csv,
                file_name='month_sale_table.csv',
                mime='text/csv',
            )



elif page == "Detailed Analysis":
    st.title("Detailed Analysis Results")

    # Áp dụng các bộ lọc trên trang phân tích chi tiết
    filtered_data = apply_filters(data, page_id="detailed_analysis")  # Pass unique page ID

    # Kiểm tra nếu filtered_data trống
    if filtered_data.empty:
        st.warning("Không có dữ liệu phù hợp với các bộ lọc đã chọn.")
    else:
        # Bảng dữ liệu có thể lọc
        st.subheader("Dữ liệu đã lọc")
        
        # Hiển thị bảng dữ liệu với tùy chọn lọc
        with st.expander("Xem Dữ Liệu"):
            st.write(filtered_data)

            # Nút tải dữ liệu
            csv_filtered_data = filtered_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải Dữ Liệu đã Lọc",
                data=csv_filtered_data,
                file_name="filtered_data.csv",
                mime="text/csv"
            )

        # Các thông tin phân tích bổ sung như đề xuất, doanh thu theo hóa đơn, khách hàng...
        selected_user_ids = st.multiselect("Chọn các User ID để nhận đề xuất:", data['CustomerID'].unique(), key="user_id_filter")

        if selected_user_ids:
            try:
                # Load pre-trained model
                model_path = "/Users/nguyenhoangvi/Downloads/NHOM2_CK/cv_model"
                best_model_loaded = CrossValidatorModel.load(model_path)
                als_model = best_model_loaded.bestModel 

                # Generate recommendations for all users
                user_recommendations_expanded = als_model.recommendForAllUsers(10) \
                    .withColumn("item_id", F.expr("explode(recommendations.item_id)")) \
                    .withColumn("rating", F.expr("explode(recommendations.rating)")) \
                    .select("user_id", "item_id", "rating")

                user_recommendations_expanded.createOrReplaceTempView("user_recommendation_table")

                # Loop through selected User IDs and fetch recommendations
                for user_id in selected_user_ids:
                    user_id_int = int(user_id)  # Convert to integer if needed

                    # Fetch recommendations for each selected user_id
                    query = f"SELECT * FROM user_recommendation_table WHERE user_id = {user_id_int}"
                    recommendations = spark.sql(query).toPandas()

                    if not recommendations.empty:
                        st.subheader(f"Đề xuất cho User ID: {user_id_int}")
                        st.write(recommendations)
                    else:
                        st.warning(f"Không tìm thấy đề xuất cho User ID: {user_id_int}")

            except ValueError:
                st.error("Vui lòng nhập các User ID hợp lệ.")

        # Thông tin phân tích tổng quan
        invoice_totals = filtered_data.groupby('InvoiceNo').agg(TotalCost=('TotalAmount', 'sum')).reset_index()

        # Doanh thu theo khách hàng
        revenue_per_customer = filtered_data.groupby('CustomerID').agg(TotalRevenue=('TotalAmount', 'sum')).reset_index()

        # Revenue by Invoice and Customer sections side by side
        col_left, col_right = st.columns(2)

        # Display "Total Revenue by Invoice" on the left side
        with col_left:
            st.subheader("Doanh thu tổng theo hóa đơn")
            st.write(invoice_totals)

            # Add download button for "Total Revenue by Invoice"
            csv_invoice_totals = invoice_totals.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải dữ liệu hóa đơn",
                data=csv_invoice_totals,
                file_name="Doanh_thu_hoa_don.csv",
                mime="text/csv"
            )

        # Display "Total Revenue by Customer" on the right side
        with col_right:
            st.subheader("Doanh thu tổng theo khách hàng")
            st.write(revenue_per_customer)

            # Add download button for "Total Revenue by Customer"
            csv_revenue_per_customer = revenue_per_customer.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Tải dữ liệu khách hàng",
                data=csv_revenue_per_customer,
                file_name="Doanh_thu_khach_hang.csv",
                mime="text/csv"
            )

        # Bảng tóm tắt phân tích
        st.subheader("Bảng tóm tắt phân tích")
        summary_df = pd.DataFrame({
            "Tổng số hóa đơn": [len(filtered_data['InvoiceNo'].unique())],
            "Tổng số khách hàng": [len(filtered_data['CustomerID'].unique())],
            "Tổng số sản phẩm": [len(filtered_data['StockCode'].unique())],
            "Tổng số lượng bán": [filtered_data['Quantity'].sum()],
            "Tổng doanh thu": [filtered_data['TotalAmount'].sum()],
            "Giá trị đơn hàng trung bình": filtered_data['TotalAmount'].mean()
        })

        st.subheader("Tóm tắt phân tích")
        st.write(summary_df)
