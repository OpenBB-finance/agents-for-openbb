FROM python:3.10-slim
WORKDIR /app

# Install required packages directly
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-dotenv==1.0.0 \
    httpx==0.25.2 \
    sse-starlette==1.8.2 \
    pdfplumber==0.10.3 \
    openbb-ai==1.7.5

# Copy the portfolio commentary application
COPY ./portfolio_commentary /app/portfolio_commentary
COPY .env /app/.env

# Expose the port the app will run on
EXPOSE 7777

# Command to run the application
CMD ["uvicorn", "portfolio_commentary.main:app", "--host", "0.0.0.0", "--port", "7777", "--reload"]