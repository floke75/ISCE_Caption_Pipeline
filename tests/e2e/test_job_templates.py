import pytest
from playwright.sync_api import Page, expect

def test_job_templates(page: Page, mock_job_list, mock_health, visual_verifier, frontend_url):
    mock_health()
    mock_job_list([])

    # 1. Load Page
    page.goto(frontend_url)

    # Wait for app to mount
    page.wait_for_selector("#root > div", timeout=10000)

    # Wait for UI
    expect(page.get_by_role("heading", name="Run inference")).to_be_visible()

    # 2. Fill fields
    locator = page.locator("label").filter(has_text="Operator notes").locator("textarea")
    expect(locator).to_be_visible()
    locator.fill("My Test Notes")

    # Select Preset "High Precision"
    page.locator("label").filter(has_text="Preset").locator("select").select_option(value="high_precision")

    # 3. Save Template
    page.get_by_role("button", name="Save current as...").click()
    page.get_by_placeholder("Template name").fill("Test Template 1")
    page.get_by_role("button", name="Save", exact=True).click()

    # Verify Toast
    expect(page.get_by_text("Template saved")).to_be_visible()

    # 4. Clear fields by reloading
    page.reload()
    expect(page.locator("label").filter(has_text="Operator notes").locator("textarea")).to_have_value("")

    # 5. Load Template
    # Select the template (it should be the 2nd option, after default)
    # Ensure options are loaded (useEffect)
    expect(page.locator("#template-select option")).to_have_count(2) # Default + 1
    page.locator("#template-select").select_option(index=1)

    # Click Load
    page.get_by_role("button", name="Load").click()

    # Verify Toast
    expect(page.get_by_text("Loaded template: Test Template 1")).to_be_visible()

    # Verify fields restored
    expect(page.locator("label").filter(has_text="Operator notes").locator("textarea")).to_have_value("My Test Notes")

    # Verify Beam Width changed to 10 (High Precision)
    expect(page.get_by_text("Beam Width (Search Depth): 10")).to_be_visible()

    # Capture Screenshot
    visual_verifier("template_system_verified")
