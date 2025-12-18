import { chromium } from 'playwright';

async function main() {
  // Use existing server or mocked environment
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 1024 } });

  // Mock API responses
  await page.route('**/api/jobs', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([
        {
          id: 'job_training_mock',
          jobType: 'training_pair',
          status: 'succeeded',
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          workspacePath: '/tmp/job_training_mock',
          params: {},
          result: {
            training_json: '/mock/training.json',
            asr_reference: '/mock/asr.json'
          }
        }
      ]),
    });
  });

  await page.route('**/api/jobs/job_training_mock/logs', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ log: 'Mock log...' }),
    });
  });

  await page.route('**/files/content?path=%2Fmock%2Ftraining.json', async (route) => {
    // Return mock tokens
    const tokens = [];
    for (let i = 0; i < 100; i++) {
        tokens.push({
            w: "word",
            start: i * 0.5,
            end: i * 0.5 + 0.4,
            pause_after_ms: i % 10 === 0 ? 600 : 50,
            break_type: i % 15 === 0 ? "LB" : "O",
            speaker_change: i % 20 === 0
        });
    }

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(tokens),
    });
  });

  await page.goto('http://localhost:5173/jobs');

  // Wait for job to appear and click it
  await page.getByText('Training pair').click();

  // Click "Data Quality" button
  await page.getByRole('button', { name: 'Data Quality' }).click();

  // Wait for dashboard to load
  await page.waitForSelector('text=Data Quality Metrics');

  // Capture screenshot
  await page.screenshot({ path: 'docs/screenshots/S14b/dashboard_implementation.png' });

  console.log('Screenshot captured.');
  await browser.close();
}

main().catch((err) => {
    console.error(err);
    process.exit(1);
});
